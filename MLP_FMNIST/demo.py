import numpy as np
import matplotlib.pyplot as plt
import time
import random

import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from sklearn.decomposition import PCA

random.seed(int(time.time()))
np.random.seed(int(time.time()))


class MLP():
    def __init__(self, train_dl, test_dl, epoch: int, learning_rate: float, gamma: float,
                 initialization="Xavier", gradient_descent_strategy="SGD",
                 data_dim=784, label_dim=10, hidden_nodes=20):
        # Gradient Descent strategy
        self.gradient_descent_strategy = gradient_descent_strategy

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma  # learning_rate decay hyperparameter gamma
        self.epoch = epoch
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.hidden_nodes = hidden_nodes
        self.initialization = initialization

        # Metrics
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

        # Dataloader
        self.train_dl = train_dl
        self.test_dl = test_dl
        # Inter Variable like z1, a1, z2
        self.inter_variable = {}

        # Gradient Descent Parameter
        self.momentum_v_layer1 = 0
        self.momentum_v_layer2 = 0
        self.momentum_beta = 0.9

        # RMSprop hyperparameters can use larger learning rate
        self.RMS_s_layer1 = 0
        self.RMS_s_layer2 = 0
        self.RMS_beta = 0.999
        self.RMS_epsilon = 1e-8

        # Adam hyperparameters
        self.Adam_v_layer1 = 0
        self.Adam_v_layer2 = 0
        self.Adam_s_layer1 = 0
        self.Adam_s_layer2 = 0
        self.Adam_beta1 = 0.9
        self.Adam_beta2 = 0.999
        self.Adam_epsilon = 1e-8

    def initialize_weights(self):
        size1 = (self.data_dim + 1, self.hidden_nodes)  # shape (785, 20), a bias weight0 at input layer
        size2 = (self.hidden_nodes, self.label_dim)  # shape (20, 10), no bias term from hidden to output layer
        if self.initialization == "Xavier":
            limit1 = np.sqrt(6 / (self.data_dim + 1 + self.hidden_nodes))
            w1 = np.random.uniform(-limit1, limit1, size1)
            limit2 = np.sqrt(6 / (self.hidden_nodes + self.label_dim))
            w2 = np.random.uniform(-limit2, limit2, size2)
        elif self.initialization == "He":
            # for ReLU (Rectified Linear Unit) activation functions
            # prevent issues such as vanishing or exploding gradients in deep learning
            stddev1 = np.sqrt(2 / (self.data_dim + 1))
            w1 = np.random.randn(*size1) * stddev1
            stddev2 = np.sqrt(2 / self.hidden_nodes)
            w2 = np.random.randn(*size2) * stddev2
        elif self.initialization == "Gaussian":
            mean = 0
            std = 0.01
            w1 = np.random.normal(mean, std, size1)
            w2 = np.random.normal(mean, std, size2)
        elif self.initialization == "Random":
            distance = 0.5
            w1 = np.random.rand(*size1) - distance / 2
            w2 = np.random.rand(*size2) + distance / 2
        elif self.initialization == "Constant0":
            # peddle with zero
            w1 = np.zeros(size1)
            w2 = np.zeros(size2)
        else:
            raise NotImplementedError("Fail to support dedicated initialization method")
        return w1, w2

    def train(self, optimizer, activation, gradient_check=False):
        start = time.time()
        w1, w2 = self.initialize_weights()

        for j in range(self.epoch):
            ema_train_accuracy = None
            ema_train_loss = None

            for step, data in enumerate(self.train_dl):  # by sample instead of batch
                learning_rate = self.learning_rate
                train_data, train_labels = data
                train_data = train_data.view(train_data.shape[0], -1).numpy().T  # shape: 784 * 500
                train_labels = F.one_hot(train_labels).numpy()  # shape: 500 * 10

                # Consider there should be a bias at the input layer
                bias_of_ones = np.ones((1, train_data.shape[1]))
                train_data = np.vstack((train_data, bias_of_ones))  # shape: 785 * 500

                if self.gradient_descent_strategy == "SGD":
                    # forward feed
                    self.forward(x=train_data, w1=w1, w2=w2, no_gradient=False,
                                 activation=activation)  # back propagation
                    # Calculate gradient
                    gradient1, gradient2 = self.back_prop(x=train_data, y=train_labels, w1=w1, w2=w2,
                                                          activation=activation)
                    w1, w2, learning_rate = self.update_weight(w1, w2, gradient1, gradient2,
                                                               optimizer=optimizer, epoch=j + 1,
                                                               learning_rate=learning_rate)
                    train_accuracy = self.accuracy(train_labels, self.inter_variable["z2"])
                    train_loss = self.loss(self.inter_variable["z2"], train_labels)

                    # Gradient check if required
                    if gradient_check:
                        self.gradient_check(train_data, train_labels, w1, w2, gradient1, gradient2,
                                            activation=activation)

                    if ema_train_accuracy is not None:
                        ema_train_accuracy = ema_train_accuracy * 0.98 + train_accuracy * 0.02
                        ema_train_loss = ema_train_loss * 0.98 + train_loss * 0.02
                    else:
                        ema_train_accuracy = train_accuracy
                        ema_train_loss = train_loss

                    if step % 50 == 0:
                        print(
                            f'Train:Step/Epoch:{step}/{j}, Accuracy:{train_accuracy * 100:.2f}, Loss:{train_loss:.4f}')
                else:
                    raise NotImplementedError("Fail to support the dedicated gradient descent strategy")

            # Evaluate
            temp_test_accuracy = []
            temp_test_loss = []
            for step, data in enumerate(self.test_dl):
                test_data, test_labels = data
                test_data = test_data.view(test_data.shape[0], -1).numpy().T
                test_labels = F.one_hot(test_labels).numpy()

                # Consider there should be a bias at the input layer
                bias_of_ones = np.ones((1, test_data.shape[1]))
                test_data = np.vstack((test_data, bias_of_ones))

                test_forward = self.forward(test_data, w1, w2, no_gradient=True, activation=activation)  # predict
                test_accuracy = self.accuracy(test_labels, test_forward)
                test_loss = self.loss(test_forward, test_labels)
                temp_test_accuracy.append(test_accuracy)
                temp_test_loss.append(test_loss)

            current_test_accuracy = np.mean(temp_test_accuracy)
            current_test_loss = np.mean(temp_test_loss)
            print(f"Epoch:{j + 1}")
            print(f"Test: Accuracy: {(100 * current_test_accuracy):.2f}%, Loss: {current_test_loss:.4f}")
            # for plot
            self.train_accuracy.append(ema_train_accuracy)
            self.train_loss.append(ema_train_loss)
            self.test_accuracy.append(current_test_accuracy)
            self.test_loss.append(current_test_loss)

        end = time.time()
        print(f"Trained time: {1000 * (end - start)} ms")

    def forward(self, x, w1, w2, no_gradient: bool, activation):
        """
        :param x: Input Data
        :param no_gradient: distinguish it's train or evaluate
        :return: if no_gradient = False, return output
        """
        # w1: (data_dim + 1, hidden_nodes), x: (data_dim + 1, batch_size); z1: (hidden_nodes, batch_size)
        z1 = w1.T.dot(x)  # w1: 785 * 20, x: 785 * 500; z1: 20 * 500

        if activation == "Sigmoid":
            a1 = 1 / (1 + np.exp(-z1))
        elif activation == "ReLU":
            a1 = np.maximum(0, z1)
        elif activation == "Tanh":
            a1 = np.tanh(z1)
        else:
            raise ValueError("Unsupported activation function")

        # w2: (hidden_nodes, label_dim), a1: (hidden_nodes, batch_size); z2: (label_dim, batch_size)
        z2 = w2.T.dot(a1)  # w2: 20 * 10, a1: 20 * 500; z2: 10 * 500

        if no_gradient:
            # for predict
            return z2
        else:
            # For back propagation
            self.inter_variable = {"z1": z1, "a1": a1, "z2": z2}

    def back_prop(self, x, y, w1, w2, activation):
        """
        :param i: for Adam bias correction
        """
        m = x.shape[1]  # x: 785 * 500, m = 500

        z1 = self.inter_variable["z1"]  # z1: 20 * 500
        a1 = self.inter_variable["a1"]  # a1: 20 * 500
        z2 = self.inter_variable["z2"]  # z2: 10 * 500

        # delta for the output layer, for all activation functions
        delta_k = z2 - y.T  # z2 - y.T, y: 500 * 10, z2: 10 * 500, delta_k: 10 * 500

        # delta for the hidden layer
        # a1: 20 * 500, w2: 20 * 10, delta_k: 10 * 500; delta_j: 20 * 500
        if activation == "Sigmoid":
            delta_j = a1 * (1 - a1) * (w2.dot(delta_k))
        elif activation == "Tanh":
            delta_j = (1 - np.power(a1, 2)) * (w2.dot(delta_k))
        elif activation == "ReLU":
            delta_j = np.where(z1 > 0, 1, 0) * (w2.dot(delta_k))
        else:
            raise ValueError("Unsupported activation function")

        # x: 785 * 500, delta_j.T: 500 * 20; gradient1: 785 * 20 (w1: 785 * 20)
        gradient1 = 1. / m * (x.dot(delta_j.T))  # 1. / m * (x dot delta_j.T)
        # a1: 20 * 500, delta_k.T: 500 * 10; gradient2: 20 * 10 (w2: 20 * 10)
        gradient2 = 1. / m * (a1.dot(delta_k.T))  # 1. / m * (a1 dot delta_k.T)
        return gradient1, gradient2

    def update_weight(self, w1, w2, gradient1, gradient2, optimizer, epoch, learning_rate):
        if optimizer == "SGD":
            return self.SGD(w1, w2, gradient1, gradient2, learning_rate)
        elif optimizer == "Momentum":
            return self.Momentum(w1, w2, gradient1, gradient2, learning_rate)
        elif optimizer == "RMSprop":
            return self.RMSprop(w1, w2, gradient1, gradient2, learning_rate)
        elif optimizer == "Adam":
            return self.Adam(epoch, w1, w2, gradient1, gradient2, learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def SGD(self, w1, w2, gradient1, gradient2, learning_rate):
        w1 = w1 - learning_rate * gradient1  # w1 - lr * g1, w1: 785 * 20, gradient1: 785 * 20
        w2 = w2 - learning_rate * gradient2  # w2 - lr * g2, w2: 20 * 10, gradient2: 20 * 10
        # Learning rate decay
        learning_rate *= self.gamma  # decay ratio, here is gamma
        return w1, w2, learning_rate

    def Momentum(self, w1, w2, gradient1, gradient2, learning_rate):
        """Exponential weighted average"""
        # Update the velocity for both layers
        self.momentum_v_layer1 = self.momentum_beta * self.momentum_v_layer1 + (1 - self.momentum_beta) * gradient1
        self.momentum_v_layer2 = self.momentum_beta * self.momentum_v_layer2 + (1 - self.momentum_beta) * gradient2

        # Update the weights
        w1 = w1 - learning_rate * self.momentum_v_layer1
        w2 = w2 - learning_rate * self.momentum_v_layer2

        # Learning rate decay
        learning_rate *= self.gamma
        return w1, w2, learning_rate

    def RMSprop(self, w1, w2, gradient1, gradient2, learning_rate):
        """Mean squared prop"""
        # Update the running average of the squared gradients
        self.RMS_s_layer1 = self.RMS_beta * self.RMS_s_layer1 + (1 - self.RMS_beta) * np.square(gradient1)
        self.RMS_s_layer2 = self.RMS_beta * self.RMS_s_layer2 + (1 - self.RMS_beta) * np.square(gradient2)

        # Update the weights
        w1 = w1 - learning_rate * gradient1 / (np.sqrt(self.RMS_s_layer1) + self.RMS_epsilon)
        w2 = w2 - learning_rate * gradient2 / (np.sqrt(self.RMS_s_layer2) + self.RMS_epsilon)

        # Learning rate decay
        learning_rate *= self.gamma
        return w1, w2, learning_rate

    def Adam(self, t, w1, w2, gradient1, gradient2, learning_rate):
        """Adaption moment estimation"""
        # Update biased first moment estimate
        self.Adam_v_layer1 = self.Adam_beta1 * self.Adam_v_layer1 + (1 - self.Adam_beta1) * gradient1
        self.Adam_v_layer2 = self.Adam_beta1 * self.Adam_v_layer2 + (1 - self.Adam_beta1) * gradient2

        # Update biased second raw moment estimate
        self.Adam_s_layer1 = self.Adam_beta2 * self.Adam_s_layer1 + (1 - self.Adam_beta2) * np.square(gradient1)
        self.Adam_s_layer2 = self.Adam_beta2 * self.Adam_s_layer2 + (1 - self.Adam_beta2) * np.square(gradient2)

        # Compute bias-corrected first moment estimate
        v_hat_layer1 = self.Adam_v_layer1 / (1 - self.Adam_beta1 ** t)
        v_hat_layer2 = self.Adam_v_layer2 / (1 - self.Adam_beta1 ** t)

        # Compute bias-corrected second raw moment estimate
        s_hat_layer1 = self.Adam_s_layer1 / (1 - self.Adam_beta2 ** t)
        s_hat_layer2 = self.Adam_s_layer2 / (1 - self.Adam_beta2 ** t)

        # Update the weights
        w1 = w1 - learning_rate * v_hat_layer1 / (np.sqrt(s_hat_layer1) + self.Adam_epsilon)
        w2 = w2 - learning_rate * v_hat_layer2 / (np.sqrt(s_hat_layer2) + self.Adam_epsilon)

        # Learning rate decay
        learning_rate *= self.gamma
        return w1, w2, learning_rate

    @staticmethod
    def accuracy(label, y_hat: np.ndarray):
        y_hat = y_hat.T
        acc = y_hat.argmax(axis=1) == label.argmax(axis=1)
        b = acc + 0
        return b.mean()

    # If there are weights1_list and weights2_list, then you can save them into file
    # def save(self, filename):
    #     np.savez(filename, self.weights1_list, self.weights2_list)

    @staticmethod
    def loss(output, label):
        # Loss = 1/n * 1/2 * ∑(yk - tk)^2
        a = label.shape[0]  # should be 10
        return np.sum(((output.T - label) ** 2)) / (2 * label.shape[0])

    def gradient_check(self, x, y, w1, w2, gradient1, gradient2, activation, epsilon=1e-7):
        parameters = np.vstack((w1.reshape((100, 1)), w2.reshape((60, 1))))
        grad = np.vstack((gradient1.reshape((100, 1)), gradient2.reshape(60, 1)))
        num_parameters = parameters.shape[0]
        gradapprox = np.zeros((num_parameters, 1))
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        for i in range(num_parameters):
            thetaplus = np.copy(parameters)
            thetaplus[i][0] = thetaplus[i][0] + epsilon
            w_plus_layer1 = thetaplus[0:100].reshape(5, 20)
            w_plus_layer2 = thetaplus[100:160].reshape(20, 3)
            J_plus[i] = self.evaluate(x, y, w_plus_layer1, w_plus_layer2, activation)

            thetaminus = np.copy(parameters)
            thetaminus[i][0] = thetaminus[i][0] - epsilon
            w_minus_layer1 = thetaminus[0:100].reshape(5, 20)
            w_minus_layer2 = thetaminus[100:160].reshape(20, 3)
            J_minus[i] = self.evaluate(x, y, w_minus_layer1, w_minus_layer2, activation)
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2. * epsilon)
        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
        difference = numerator / denominator
        print(f"L2 distance of Gradient check:{difference}")

    def evaluate(self, x, y, w1, w2, activation):
        z1 = w1.T.dot(x)  # w1: 785 * 20, x: 785 * 500, z1: 20 * 500
        if activation == "Sigmoid":
            a1 = 1 / (1 + np.exp(-z1))
        elif activation == "ReLU":
            a1 = np.maximum(0, z1)
        elif activation == "Tanh":
            a1 = np.tanh(z1)
        else:
            raise ValueError("Unsupported activation function")
        z2 = w2.T.dot(a1)  # w2: 20 * 10, a1: 20 * 500, z2: 10 * 500
        return np.sum(((z2.T - y) ** 2) / (2 * y.shape[0]))

    def plot_test(self):
        plt.figure(figsize=(7, 6))
        plt.xlabel(f"Epochs({self.epoch} Epoch)")
        plt.ylabel("Accuracy")
        plt.plot(self.test_accuracy, label="Test Accuracy", alpha=0.5)
        plt.xticks(np.arange(0, len(self.test_accuracy)))
        plt.legend()
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(7, 6))
        plt.xlabel("Epochs(10 Epoch/step)")
        plt.ylabel("Loss")
        plt.plot(np.array(self.train_loss), label="Train Loss", alpha=0.5)
        plt.xticks(np.arange(0, len(self.train_loss)))
        plt.legend()
        plt.show()


def prepare_pca_transform(train_data, n_components):
    # Combine all training data into a single numpy array
    data = []
    for img, _ in train_data:
        data.append(img.numpy().flatten())
    data = np.array(data)

    # Fit PCA with whitening and dimensionality reduction
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(data)

    return pca


def transform_dataset(dataset, pca):
    data = []
    labels = []
    for img, label in dataset:
        data.append(img.numpy().flatten())
        labels.append(label)

    data = np.array(data)
    data_pca = pca.transform(data)
    labels = np.array(labels)

    data_pca_tensor = torch.tensor(data_pca, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return torch.utils.data.TensorDataset(data_pca_tensor, labels_tensor)


if __name__ == '__main__':
    # Hyperparameters
    epoch = 50                      # 20, 30, 50, 100, so on, start with a smaller number
    gamma = 0.99                    # (0.95, 0.99)
    initialization = "Xavier"       # "Xavier", "He", "Gaussian", "Random", "Constant0" (first two are better)
    optimizer = "Adam"              # "SGD", "Momentum", "RMSprop", "Adam" ("Adam combines Momentum and RMSprop")
    activation = "ReLU"             # "Sigmoid", "ReLU", "Tanh" (ReLu performs well, Sigmoid may work well here)
    batch_size = 500                # typical value 32, 64, 128
    n_components = 0.98             # Number of PCA components for dimensionality reduction

    learning_rate_list = [0.0005, 0.001, 0.005]
    hidden_nodes_list = [100, 200, 300]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # normalize
    train_set = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)

    # Fit PCA on the training data
    pca = prepare_pca_transform(train_set, n_components)
    input_nodes = pca.n_components_
    print(f"The PCA correlation ratio is {sum(pca.explained_variance_ratio_)}")
    print(f"The dimension of input layer is {input_nodes}")

    # Transform the training and test datasets
    train_set_pca = transform_dataset(train_set, pca)
    test_set_pca = transform_dataset(test_set, pca)

    train_dl = torch.utils.data.DataLoader(train_set_pca, batch_size=batch_size, drop_last=True, num_workers=4,
                                           shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_set_pca, batch_size=batch_size, drop_last=False, num_workers=4,
                                          shuffle=True)

    # Store the accuracies for each combination of learning rate and hidden nodes
    accuracies = {}

    for learning_rate in learning_rate_list:
        for hidden_nodes in hidden_nodes_list:
            print(f"\nTraining with learning_rate={learning_rate} and hidden_nodes={hidden_nodes}")
            mlp = MLP(train_dl, test_dl, epoch, learning_rate, gamma, initialization, "SGD", input_nodes, 10,
                      hidden_nodes)
            mlp.train(optimizer, activation, False)

            # Store the final test accuracy for this combination
            final_test_accuracy = mlp.test_accuracy[-1]
            accuracies[(learning_rate, hidden_nodes)] = final_test_accuracy

            print(f"After Epoch {epoch} - Train Loss: {mlp.train_loss[-1]:.4f} \
                  - Train Accuracy: {mlp.train_accuracy[-1] * 100:.2f} \
                  - Test Loss: {mlp.test_loss[-1]:.4f} \
                  - Test Accuracy: {final_test_accuracy * 100:.2f}")

    # print out the accuracies
    print(accuracies)

    # Plot accuracy results for each hidden node configuration
    plt.figure(figsize=(10, 6))
    for hidden_nodes in hidden_nodes_list:
        acc_values = [accuracies[(lr, hidden_nodes)] for lr in learning_rate_list]
        plt.plot(learning_rate_list, acc_values, marker='o', label=f'Hidden Nodes={hidden_nodes}')

    plt.title('MLP Accuracy vs Learning Rate for Different Hidden Node Configurations on Fashion-MNIST with PCA')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()
