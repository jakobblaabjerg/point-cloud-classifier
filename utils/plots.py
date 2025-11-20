import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc

def plot_data(dataset, sample_size=None, random_state=42, save_dir=None):

    df = dataset.copy()

    if sample_size is not None:
        df = df.groupby("label", group_keys=False).sample(
            n=sample_size, random_state=random_state
        )

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x="energy_total", 
        y="hits_total", 
        hue="label", 
        alpha=0.7
    )
    plt.xlabel("Shower Energy (MeV)")
    plt.ylabel("Number of Hits")
    plt.legend(title="Particle")
    plt.grid(True)

    if save_dir:
        save_path = os.path.join(save_dir, "plot.png")
        plt.savefig(save_path)
        plt.close()   
    
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, save_dir=None, split_name="test"):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix ({split_name})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f"confusion_matrix_{split_name}.png")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(y_true, y_prob, save_dir=None, split_name="test"):

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title(f"ROC Curve ({split_name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, f"roc_curve_{split_name}.png")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 


def plot_precision_recall_curve(y_true, y_prob, save_dir=None, split_name="test"):

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    plt.title(f"Precision-Recall Curve ({split_name})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f"precision_recall_{split_name}.png")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()