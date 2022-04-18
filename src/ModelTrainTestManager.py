"""
File adapted from :
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

from tabnanny import verbose
from turtle import distance
from cv2 import threshold
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
from typing import Callable, Type
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

from src.utils import accuracy
from src.DataManager import DataManager


class ModelTrainTestManager(object):
    """
    Class used the train and test the given model in the parameters
    """

    def __init__(self, model:torch.nn.Module,
                 data_manager: DataManager,
                 loss_fn: torch.nn.Module,
                 optimizer_factory: (Callable[[torch.nn.Module],
                                              torch.optim.Optimizer]),
                 exp_name:str,
                 accuracy_mesure: (Callable[[torch.Tensor, torch.Tensor], 
                                            float])=accuracy,
                 learning_rate:float=0.01,
                 use_cuda:bool=False,
                 verbose:bool=True,
        ):
        """CNNTrainTestManager object creation

        Args:
            model (torch.nn.Module): model to train
            data_manager (DataManager): data manager used to get training, testing and validation loaders
            loss_fn (torch.nn.Module): the loss function used
            optimizer_factory (Callable[[torch.nn.Module], torch.optim.Optimizer]): A callable to create the optimizer
            exp_name (str): Experiment name, define in which directory the model will be stored
            accuracy_mesure (Callable[[torch.Tensor, torch.Tensor], float], optional): Specify the accuracy mesure to use. Defaults to None.
            learning_rate (float, optional): Learning rate used during the training. Defaults to 0.01.
            use_cuda (bool, optional): to Use the gpu to train the model. Defaults to False.
            verbose (bool, optional): _description_. Defaults to True.
        """
        

        device_name = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Suppress this warning by"
                          "passing use_cuda=False to {}()."
                          .format(self.__class__.__name__), RuntimeWarning)
            device_name = 'cpu'

        self.device = torch.device(device_name)
        self.data_manager = data_manager
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer_factory(self.model)
        self.model = self.model.to(self.device)
        self.use_cuda = use_cuda
        self.metric_values = {}
        self.learning_rate = learning_rate
        self.exp_name = exp_name
        self.verbose = verbose
        self.accuracy_mesure = accuracy_mesure
        batch_size = data_manager.get_train_loader().batch_size
        self.train_embeddings = np.zeros((len(data_manager.get_train_loader()) * batch_size, model.embedding_size))
        self.val_embeddings = np.zeros((len(data_manager.get_validation_loader()) * batch_size, model.embedding_size))
        self.test_embeddings = np.zeros((len(data_manager.get_test_loader()) * batch_size, model.embedding_size))

    def train(self, num_epochs:int, start_epoch:int=0, metric_values:dict=None):
        """Train the model for a given number of epochs

        Args:
            num_epochs (int): Number of epochs to do in total
            Args below are useful to resume training of an already trained model 
            start_epoch (int, optional): The number of epochs already done during previous training. Defaults to 0.
            metric_values (dict, optional): Metric values obtained during previous training. Defaults to None.
        """
        # Initialize metrics container
        if metric_values is not None:
            self.metric_values = metric_values
        else:
            self.metric_values['train_loss'] = []
            self.metric_values['train_acc'] = []
            self.metric_values['val_loss'] = []
            self.metric_values['val_acc'] = []

        # Create pytorch's train data_loader
        train_loader = self.data_manager.get_train_loader()

        # train num_epochs times
        for epoch in range(start_epoch, num_epochs):
            if self.verbose:
                print("Epoch: {} of {}".format(epoch + 1, num_epochs))
            train_loss = 0.0

            with tqdm(range(len(train_loader))) as t:
                train_losses = []
                train_accuracies = []
                for i, data in enumerate(train_loader, 0):
                    # transfer tensors to selected device
                    train_inputs, train_labels = data[0].to(self.device, dtype=torch.float), \
                                                 data[1].to(self.device, dtype=torch.long)
                                                 
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    train_outputs, emb = self.model(train_inputs, train_labels)
                    # computes loss using loss function loss_fn
                    loss = self.loss_fn(train_outputs, train_labels)
                    # for croosentropy loss softmax and argmax not needed :
                    # "The input is expected to contain raw, unnormalized scores for each class"

                    # Use autograd to compute the backward pass.
                    loss.backward()

                    # updates the weights using gradient descent                    
                    self.optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(loss.item())
                    train_accuracies.append(
                        self.accuracy(train_outputs, train_labels))

                    # print metrics along progress bar
                    train_loss += loss.item()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()

                    # saving the embedding
                    self.train_embeddings[i:i+emb.size(0),:] = emb.detach().cpu().numpy()

            # evaluate the model on validation data after each epoch
            self.metric_values['train_loss'].append(np.mean(train_losses))
            self.metric_values['train_acc'].append(np.mean(train_accuracies))
            self.evaluate_on_validation_set()

        print("Finished training.")

    
    def evaluate_on_validation_set(self):
        """
        function that evaluate the model on the validation set every epoch
        """
        # switch to eval mode so that layers like batchnorm's layers nor
        # dropout's layers works in eval mode instead of training mode
        self.model.eval()

        # Get validation data
        val_loader = self.data_manager.get_validation_loader()
        validation_loss = 0.0
        validation_losses = []
        validation_accuracies = []

        with torch.no_grad():
            for j, val_data in enumerate(val_loader, 0):
                # transfer tensors to the selected device
                val_inputs, val_labels = val_data[0].to(self.device, dtype=torch.float), \
                                         val_data[1].to(self.device, dtype=torch.long)

                # forward pass
                val_outputs, emb = self.model(val_inputs,val_labels)

                # compute loss function
                loss = self.loss_fn(val_outputs, val_labels)
                validation_losses.append(loss.item())
                validation_accuracies.append(
                    self.accuracy(val_outputs, val_labels))
                validation_loss += loss.item()

                # save the embeddings
                self.val_embeddings[j:j+emb.size(0),:] = emb.detach().cpu().numpy()

        self.metric_values['val_loss'].append(np.mean(validation_losses))
        self.metric_values['val_acc'].append(np.mean(validation_accuracies))

        # displays metrics
        if self.verbose:
            print('Validation loss %.3f' % (validation_loss / len(val_loader) *100 ))

        # switch back to train mode
        self.model.train()

    def accuracy(self, outputs, labels):
        """
        Computes the accuracy of the model
        Args:
            outputs: outputs predicted by the model
            labels: real outputs of the data
        Returns:
            Accuracy of the model
        """
        return self.accuracy_mesure(outputs, labels)

    def evaluate_on_test_set(self):
        """
        Evaluate the model on the test set
        :returns;
            Accuracy of the model on the test set
        """
        test_loader = self.data_manager.get_test_set()
        accuracies = 0
        with torch.no_grad():
            for data in test_loader:
                test_inputs, test_labels = data[0].to(self.device, dtype=torch.float), \
                                           data[1].to(self.device, dtype=torch.long)
                test_outputs = self.model(test_inputs,test_labels)
                assert torch.where(test_labels)
                accuracies += self.accuracy(test_outputs, test_labels)
        print("Accuracy on the test set: {:05.3f} %".format(
            accuracies / len(test_loader)))

    def get_embeddings_on_test_set(self):

        test_loader = self.data_manager.get_test_loader()
        accuracies = 0
        with torch.no_grad():
            for j,data in enumerate(test_loader,0):
                test_inputs, test_labels = data[0].to(self.device, dtype=torch.float), \
                                           data[1].to(self.device, dtype=torch.long)
                test_outputs, emb = self.model(test_inputs,test_labels)
                assert torch.where(test_labels)
                accuracies += self.accuracy(test_outputs, test_labels)
                self.test_embeddings[j:j+emb.size(0),:] = emb.detach().cpu().numpy()

    def inference_for_test_set(self, threshold=0.7):
        known_embeddings = np.concatenate((self.train_embeddings, self.val_embeddings))
        known_labels = np.concatenate((self.data_manager.get_train_set()["individual_id"].to_numpy(), self.data_manager.get_train_set()["individual_id"].to_numpy()))
        # use nearest neighbors to determine the labels in test set 
        nn = NearestNeighbors(n_neighbors=100, metric="cosine")
        nn.fit(known_embeddings)
        distances,idxs = nn.kneighbors(self.test_embeddings, return_distance=True)
        conf = 1-distances # we have to maximize conf to get the more plausible labels

        # pour chaque test
        for i in range(len(self.data_manager.get_test_set())):
            '''
            print("conf ",i,"eme image test: ", conf[i,:])
            sorted_index = np.argsort(-conf[i, :])
            print("sorted_index: ", sorted_index)
            sorted_array = conf[i,:][sorted_index] # sort the array for test_embedding i
            print("sorted_array: ", sorted_array)'''

            # conf est déjà trié
            sorted_array = conf[i,:]

            #train_indexes = idxs[i, sorted_index]
            train_indexes = idxs[i]
            sorted_labels = known_labels[train_indexes]

            indexes = np.unique(sorted_labels, return_index=True)[1]
            sorted_index = np.sort(indexes)

            top5_indexes = sorted_index[: min(5, len(sorted_index))] # top5 (or less if there is too many duplicates)
            top5_conf = sorted_array[top5_indexes]
            top5_labels = sorted_labels[top5_indexes]

            top5_conf = np.concatenate((top5_conf, [-1]))
            index_below_threshold = np.argmax(top5_conf < threshold)

            top5_label_ids = self.data_manager.get_individual_id(top5_labels)

            top5_lbl_newId = np.concatenate((top5_label_ids[:index_below_threshold], ['new_individual'], top5_label_ids[index_below_threshold:]))

            top5_lbl_newId= top5_lbl_newId[: min(5, len(top5_lbl_newId))]


            self.data_manager.get_test_set()["predictions"].iloc[i] = " ".join(top5_lbl_newId)
    
    def saveTest(self, test_file):
        self.data_manager.get_test_set().to_csv(path_or_buf=test_file, columns=["image", "predictions"])


    def plot_metrics(self, path):
        """
        Function that plots train and validation losses and accuracies after
        training phase
        """
        epochs = range(1, len(self.metric_values['train_loss']) + 1)

        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(
            epochs, self.metric_values['train_loss'],
            '-o', label='Training loss')
        ax1.plot(
            epochs, self.metric_values['val_loss'],
            '-o', label='Validation loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # accuracy plot
        ax2.plot(
            epochs, self.metric_values['train_acc'], '-o',
            label='Training accuracy')
        ax2.plot(
            epochs, self.metric_values['val_acc'], '-o',
            label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        #f.savefig(join(path, 'fig1.png'))
        plt.show()

def optimizer_setup(
        optimizer_class: Type[torch.optim.Optimizer],
        **hyperparameters
    ) -> Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the
    given hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an
    argument.

    Args:
        optimizer_class: optimizer used to train the model
        **hyperparameters: hyperparameters for the model
        Returns:
            function to setup the optimizer
    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f