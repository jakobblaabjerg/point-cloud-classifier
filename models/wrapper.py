import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm


class ModelWrapper:

    def __init__(self, 
                 model,
                 learning_rate, 
                 epochs, 
                 log_dir
                 ):
        
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.log_dir = log_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = 10
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.checkpoint_path = os.path.join(self.log_dir, "best_model.pt") if log_dir else None
        self.model.to(self.device)

    def fit(self, train_loader, val_loader=None):

        criterion = nn.BCEWithLogitsLoss() 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        writer = SummaryWriter(self.log_dir)

        for epoch in range(self.epochs):
            self.model.train()
            batch_losses = []
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for batch in loop:

                if len(batch) == 2:
                    batch_X, batch_y = batch
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    logits = self.model(batch_X)

                elif len(batch) == 3:
                    batch_X, batch_aux, batch_y = batch
                    batch_X = batch_X.to(self.device)
                    batch_aux = batch_aux.to(self.device)
                    batch_y = batch_y.to(self.device)
                    logits = self.model(batch_X, batch_aux)

                # print("logits:", logits[:5].detach().cpu().numpy().flatten())
                # print("logits mean:", logits.mean().item(), "std:", logits.std().item())
                # print("labels mean:", batch_y.mean().item())
                # print("X mean:", batch_X.mean().item(), "std:", batch_X.std().item())

                # if len(batch) == 3:
                #     # Sparse batching — check counts
                #     print("aux shape:", batch_aux.shape)
                #     print("aux unique:", torch.unique(batch_aux))


                # print(torch.unique(batch_y))
                # print(logits.shape, batch_y.shape)
                # print(batch_y.mean())
                # break
                optimizer.zero_grad()
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                loop.set_postfix(loss=loss.item())

            epoch_loss = sum(batch_losses) / len(batch_losses)
            writer.add_scalar("Loss/train", epoch_loss, epoch)

            # y_true_train, y_pred_train = self.predict(train_loader)
            # train_acc = (y_true_train == y_pred_train).mean()
            # writer.add_scalar("Accuracy/train", train_acc, epoch)

            print("logits:", logits[:5].detach().cpu().numpy().flatten())
            print("logits mean:", logits.mean().item(), "std:", logits.std().item())
            print("labels mean:", batch_y.mean().item())
            print("X mean:", batch_X.mean().item(), "std:", batch_X.std().item())

            if len(batch) == 3:
                # Sparse batching — check counts
                print("aux shape:", batch_aux.shape)
                print("aux unique:", torch.unique(batch_aux))


            if val_loader:

                self.model.eval()
                val_losses = []
                y_true_val, y_pred_val = [], []


                with torch.no_grad():
                    for batch in val_loader:

                        if len(batch) == 2:
                            batch_X, batch_y = batch
                            batch_X = batch_X.to(self.device)
                            batch_y = batch_y.to(self.device)
                            logits = self.model(batch_X)

                        elif len(batch) == 3:
                            batch_X, batch_aux, batch_y = batch
                            batch_X = batch_X.to(self.device)
                            batch_aux = batch_aux.to(self.device)
                            batch_y = batch_y.to(self.device)
                            logits = self.model(batch_X, batch_aux)

                        loss = criterion(logits, batch_y)
                        val_losses.append(loss.item())

                        probs = torch.sigmoid(logits)
                        preds = (probs >= 0.5).float()
                        y_true_val.append(batch_y.cpu())
                        y_pred_val.append(preds.cpu())

                val_loss = sum(val_losses) / len(val_losses)
                y_true_val = torch.cat(y_true_val, dim=0).numpy()
                y_pred_val = torch.cat(y_pred_val, dim=0).numpy()
                val_acc = (y_true_val == y_pred_val).mean()

                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Accuracy/val", val_acc, epoch)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    print(f"Epoch {epoch+1}: New best model saved (val_loss={val_loss:.4f})")
                else:
                    self.early_stop_counter += 1
                    print(f"Epoch {epoch+1}: No improvement ({self.early_stop_counter}/{self.patience})")

                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        writer.close()



    def predict(self, data_loader, return_prob=False):

        self.model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in data_loader:

                if len(batch) == 2:
                    batch_X, batch_y = batch
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    logits = self.model(batch_X)

                elif len(batch) == 3:
                    batch_X, batch_aux, batch_y = batch
                    batch_X = batch_X.to(self.device)
                    batch_aux = batch_aux.to(self.device)
                    batch_y = batch_y.to(self.device)
                    logits = self.model(batch_X, batch_aux)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                y_true.append(batch_y.cpu())
                y_pred.append(preds.cpu())
                y_prob.append(probs.cpu())
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        y_prob = torch.cat(y_prob).numpy()

        if return_prob:
            return y_true, y_prob
        else:
            return y_true, y_pred

    def save(self, save_dir):

        model_path = os.path.join(save_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)

    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)