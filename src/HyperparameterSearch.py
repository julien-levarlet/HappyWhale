from typing import Callable
import torch
from src.ModelTrainTestManager import ModelTrainTestManager
from src.DataManager import DataManager
from src.Models.HappyWhaleModel import HappyWhaleModel
from src.utils import accuracy

class HyperparameterSearchManager():
    """
    This class allows to do an hyperparameter search, to find the best model on the validation set
    """

    def __init__(self, model_class, 
                data_manager:DataManager, 
                params:dict,
                loss_fn: torch.nn.Module,
                optimizer_factory: (Callable[[torch.nn.Module],
                                              torch.optim.Optimizer]),
                exp_name:str,
                accuracy_mesure: (Callable[[torch.Tensor, torch.Tensor], 
                                            float])=accuracy,
                num_epoch=3,
        ) -> None:
        """
        Args:
            model_class : The model we want to train
            data_manager (DataManager):  The data manager that contains the train and validation sets for the hyperparameter search
            params (dict): Values of learning rate, scale and margin to iterate for the search
            loss_fn (torch.nn.Module): The loss used
            optimizer_factory (Callable[[torch.nn.Module], torch.optim.Optimizer]): A callable to create the optimizer
            exp_name (str): Experiment name, define in which directory the model will be stored
            accuracy_mesure (Callable[[torch.Tensor, torch.Tensor], float], optional): Specify the accuracy mesure to use. Defaults to accuracy.
            num_epoch (int, optional): Number of epochs for each training. Defaults to 3.
        """        
        
        self.model_class = model_class
        self.best_model_trainer = None
        self.best_param = None
        self.best_val_acc = 0
        self.exp_name = exp_name


        for lr in params["learning_rate"]:
            for s in params["arcface_s"]:
                for m in params["arcface_m"]:
                    print("Parameters : lr", lr, ", s", s, ", m", m)
                    arcFace_config = {
                        "s": s,  # scale (The scale parameter changes the shape of the logits. The higher the scale, the more peaky the logits vector becomes.)
                        "m": m,  # margin (margin results in a bigger separation of classes in your training set)
                        "ls_eps": 0.0,
                        "easy_margin": False
                    }
                    model = HappyWhaleModel("tf_efficientnet_b0_ns", 512, num_class=20, arcface_config=arcFace_config)
                    train_manager = ModelTrainTestManager(model, data_manager, loss_fn, optimizer_factory, exp_name, accuracy_mesure, lr, use_cuda=True, verbose=False)
                    train_manager.train(num_epoch)

                    val_loss = train_manager.metric_values["val_loss"][-1]
                    val_acc = train_manager.metric_values["val_acc"][-1] # we sort model based on final val accuracy
                    print("Validation loss :", val_loss, ", validation accuracy :", val_acc)
                    
                    if val_acc >= self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_model_trainer = train_manager
                        self.best_param = {
                            "learning_rate":lr,
                            "arcface_s":s,
                            "arcface_m":m,
                        }
                    
    def evaluate_best_on_test_set(self):
        """
        Function used to evaluate the best model after the search
        """
        self.best_model_trainer.evaluate_on_test_set()
        self.best_model_trainer.plot_metrics(self.exp_name)


    def get_best_model(self):
        return self.best_model_trainer.model
    def get_best_param(self):
        return self.best_param