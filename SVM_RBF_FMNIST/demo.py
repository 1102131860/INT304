import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time  # Import time module

# Load Fashion-MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

X_train = train_set.data.numpy().reshape(-1, 28*28)
y_train = train_set.targets.numpy()
X_test = test_set.data.numpy().reshape(-1, 28*28)
y_test = test_set.targets.numpy()

# Preprocess data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
n_components = 0.98  # Retain 98% of variance
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define function to train SVM with given C and gamma values and measure training time
def train_svm(C, gamma):
    print(f"Training SVM with C={C} and gamma={gamma}")
    start_time = time.time()  # Start timing
    model = SVC(C=C, kernel='rbf', gamma=gamma, max_iter=1000000)
    model.fit(X_train_pca, y_train)
    end_time = time.time()  # End timing
    training_time = end_time - start_time  # Calculate elapsed time
    y_pred = model.predict(X_test_pca)
    score = accuracy_score(y_test, y_pred)
    print(f"SVM with C={C} and gamma={gamma} has accuracy: {score} and training time: {training_time:.2f} seconds")
    return score, training_time

# List of C and gamma values to try
C_values = [0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1]

# Parallel training of multiple SVM models and collecting results
results = Parallel(n_jobs=-1)(delayed(train_svm)(C, gamma) for C in C_values for gamma in gamma_values)

# Output results
print("Training completed.")
for (C, gamma), (accuracy, training_time) in zip([(C, gamma) for C in C_values for gamma in gamma_values], results):
    print(f"C={C}, gamma={gamma}: Accuracy={accuracy}, Training Time={training_time:.2f} seconds")

# Separate accuracy and training time results for plotting
accuracies, training_times = zip(*results)

# Reshape the accuracies and training times for plotting
accuracies = np.array(accuracies).reshape(len(C_values), len(gamma_values))
training_times = np.array(training_times).reshape(len(C_values), len(gamma_values))

# Plot accuracy results
plt.figure(figsize=(10, 6))
for i, C in enumerate(C_values):
    plt.plot(gamma_values, accuracies[i], marker='o', label=f'C={C}')
plt.title('SVM Accuracy vs. Gamma Values for Different C Values on Fashion-MNIST with PCA')
plt.xlabel('Gamma Value')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.grid(True)
plt.legend()
plt.show()
