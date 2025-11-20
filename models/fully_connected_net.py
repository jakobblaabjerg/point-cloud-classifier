import torch 
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

class FullyConnectedNet(nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_layers, 
                 output_dim,
                 epochs,
                 learning_rate,
                 log_dir,
                 ):

        super().__init__()
        layers = []
        in_features = input_dim
        for hidden in hidden_layers:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden
        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = 10
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.checkpoint_path = os.path.join(self.log_dir, "best_model.pt")

    def forward(self, x):
        logits = self.network(x)
        return logits
    

    def fit(self, train_loader, val_loader):

        writer = SummaryWriter(self.log_dir)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)
        
        for epoch in range(self.epochs):
            self.train()
            batch_losses = []
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for batch_X, batch_y in loop:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.float().to(self.device)
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                loop.set_postfix(loss=loss.item())

            epoch_loss = sum(batch_losses) / len(batch_losses)
            writer.add_scalar("Loss/train", epoch_loss, epoch)

            y_true_train, y_pred_train = self.predict(train_loader)
            train_acc = (y_true_train == y_pred_train).mean()
            writer.add_scalar("Accuracy/train", train_acc, epoch)

            if val_loader is not None:

                self.eval()
                val_losses = []
                y_true_val, y_pred_val = [], []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device) 
                        batch_y = batch_y.to(self.device)
                        outputs = self(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_losses.append(loss.item())

                        probs = torch.sigmoid(outputs)
                        preds = (probs >= 0.5).float()
                        y_true_val.append(batch_y.cpu())
                        y_pred_val.append(preds.cpu())

                val_loss = sum(val_losses) / len(val_losses)
                writer.add_scalar("Loss/val", val_loss, epoch)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                    torch.save(self.state_dict(), self.checkpoint_path)
                    print(f"Epoch {epoch+1}: New best model saved (val_loss={val_loss:.4f})")
                else:
                    self.early_stop_counter += 1
                    print(f"Epoch {epoch+1}: No improvement ({self.early_stop_counter}/{self.patience})")

                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

                y_true_val = torch.cat(y_true_val, dim=0).numpy()
                y_pred_val = torch.cat(y_pred_val, dim=0).numpy()
                val_acc = (y_true_val == y_pred_val).mean()
                writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.close()


    def predict(self, data_loader, return_prob=False):
        self.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self(batch_X)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                y_true.append(batch_y.cpu()) 
                y_pred.append(preds.cpu())
                y_prob.append(probs.cpu())
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        y_prob = torch.cat(y_prob, dim=0).numpy()

        if return_prob:
            return y_true, y_prob
        else:
            return y_true, y_pred
    
    def save(self, save_dir):
        model_path = os.path.join(save_dir, "model.pt")
        torch.save(self.state_dict(), model_path)



#check point and early stopping