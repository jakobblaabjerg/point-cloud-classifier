from utils.data import Step2PointGraph, Step2PointPointCloud, Step2PointTabular
from utils.log import TrainingLogger
from utils.config import load_config, save_config
from utils.plots import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from models.logistic_regression import LogRegression
from models.fully_connected_net import FullyConnectedNet
from models.wrapper import ModelWrapper
from models.graph_net import GraphNet
from models.deep_sets import DeepSets
import numpy as np 
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report
import torch
import json

def get_dataloader(dataset_name, config):

    if dataset_name == "s2pt":
        dataloader = Step2PointTabular(**config["dataset"])
    elif dataset_name == "s2ppc":
        dataloader = Step2PointPointCloud(**config["dataset"])
    elif dataset_name == "s2pg":
        dataloader = Step2PointGraph(**config["dataset"])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataloader


def get_model(model_name, config, model_dir=None):

    if model_name == "logistic_regression":

        model = LogRegression()

        if model_dir is not None:
            model_path = os.path.join(model_dir, "model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"LogisticRegression model not found at {model_path}")
            with open(model_path, "rb") as f:
                model.clf = pickle.load(f)   
            print(f"Loaded LogisticRegression model from {model_path}")
           
    elif model_name == "fully_connected_net":

        model = FullyConnectedNet(**config["model"]) 
        model = ModelWrapper(model, **config["trainer"], **config["logging"])

        if model_dir is not None:
            model_path = os.path.join(model_dir, "best_model.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"FullyConnectedNet model not found at {model_path}")
            model.load(model_path)
            print(f"Loaded FullyConnectedNet model from {model_path}")             
    

    elif model_name == "deep_sets":

        model = DeepSets(**config["model"]) 
        model = ModelWrapper(model, **config["trainer"], **config["logging"])

        if model_dir is not None:
            model_path = os.path.join(model_dir, "best_model.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"DeepSets model not found at {model_path}")
            model.load()
            print(f"Loaded DeepSets model from {model_path}")    


    elif model_name == "graph_net":

        model = GraphNet(**config["model"]) 
        model = ModelWrapper(model, **config["trainer"], **config["logging"])

        if model_dir is not None:
            model_path = os.path.join(model_dir, "model.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"GraphNet model not found at {model_path}")
            model.load()
            print(f"Loaded GraphNet model from {model_path}")   

    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def evaluate_model(model_dir, save_dir):

    config_path = os.path.join(model_dir, "config.yaml")
    config = load_config(config_path)

    model_name = config["meta"]["model_name"]
    dataset_name = config["meta"]["dataset_name"]

    dataloader = get_dataloader(dataset_name=dataset_name, config=config)
    model = get_model(model_name=model_name, config=config, model_dir=model_dir)

    test_loader = dataloader.get_test_loader()
    y_true_test, y_pred_test = model.predict(test_loader)

    acc_test = accuracy_score(y_true_test, y_pred_test)
    print("accuracy/test", round(acc_test, 6))

    train_loader = dataloader.get_train_loader()
    y_true_train, y_pred_train = model.predict(train_loader)

    acc_train = accuracy_score(y_true_train, y_pred_train)
    print("accuracy/train", round(acc_train, 6))

    val_loader = dataloader.get_val_loader()
    y_true_val, y_pred_val = model.predict(val_loader)

    acc_val = accuracy_score(y_true_val, y_pred_val)
    print("accuracy/val", round(acc_val, 6))


    metrics = {
        "accuracy_train": float(acc_train),
        "accuracy_val": float(acc_val),
        "accuracy_test": float(acc_test),
    }

    save_path = os.path.join(save_dir, "metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)



    classification_report(y_true_test, y_pred_test)
    report = classification_report(y_true_test, y_pred_test)
    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    y_true_test, y_prob_test = model.predict(test_loader, return_prob=True)
    plot_confusion_matrix(y_true_test, y_pred_test, save_dir)
    plot_precision_recall_curve(y_true_test, y_prob_test, save_dir)
    plot_roc_curve(y_true_test, y_prob_test, save_dir)



def train_model(model_name: str, dataset_name: str, config, plots=False, return_log_dir=False):

    dataset_name = dataset_name.lower()
    model_name = model_name.lower()

    # setup log
    logger = TrainingLogger(model_name, dataset_name, **config["logging"])
    version = logger.get_version()
    log_dir = os.path.join(config["logging"]["log_dir"], f"version_{version}")
    config["logging"]["log_dir"] = log_dir
    config["meta"]["model_name"] = model_name
    config["meta"]["dataset_name"] = dataset_name 

    dataloader = get_dataloader(dataset_name=dataset_name, config=config)
    model = get_model(model_name=model_name, config=config)

    train_loader = dataloader.get_train_loader()
    val_loader = dataloader.get_val_loader()

    # fit model
    save_config(config=config, log_dir=log_dir)
    model.fit(train_loader, val_loader)
    model.save(save_dir=log_dir) 
    
    # predict 
    y_true_train, y_pred_train = model.predict(train_loader)
    y_true_val, y_pred_val = model.predict(val_loader)

    # evaluate performance
    acc_train = accuracy_score(y_true_train, y_pred_train)
    acc_val = accuracy_score(y_true_val, y_pred_val)
    logger.log_metric("accuracy/train", round(acc_train, 6))
    logger.log_metric("accuracy/val", round(acc_val, 6))
    logger.log_metric("parameters", model.get_trainable_parameters())

    if plots:
        y_true_val, y_prob_val = model.predict(val_loader, return_prob=True)
        plot_confusion_matrix(y_true_val, y_pred_val, log_dir)
        plot_precision_recall_curve(y_true_val, y_prob_val, log_dir)
        plot_roc_curve(y_true_val, y_prob_val, log_dir)

    if return_log_dir:
        return log_dir
    return None

if __name__ == "__main__": 

    # model = "fully_connected_net"
    # #model = "logistic_regression"
    # #model = "graph_net"
    # dataset = "s2pt" 
    # config = load_config("configs/base.yaml", f"configs/{model}.yaml")
    # train_model(model, dataset, config, plots=True)


    version = 99
    model_dir = f"log/version_{version}"
    save_dir = f"results/version_{version}"
    os.makedirs(save_dir, exist_ok=True)
    evaluate_model(model_dir=model_dir, save_dir=save_dir)