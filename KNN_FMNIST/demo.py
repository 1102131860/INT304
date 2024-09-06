import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# Step 1: Load the dataset
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)


# Convert to numpy arrays
def convert_to_numpy(dataset):
    data = []
    targets = []
    for img, label in dataset:
        data.append(img.numpy().flatten())
        targets.append(label)
    return np.array(data), np.array(targets)


train_data, train_labels = convert_to_numpy(train_set)
test_data, test_labels = convert_to_numpy(test_set)

# Step 2: Standardize the data
scaler = StandardScaler().fit(train_data)
train_data_standardized = scaler.transform(train_data)
test_data_standardized = scaler.transform(test_data)

# Step 3: Apply PCA (optional)
n_components = 0.98  # Retain 98% of variance
pca = PCA(n_components=n_components)
train_data_pca = pca.fit_transform(train_data_standardized)
test_data_pca = pca.transform(test_data_standardized)

# Step 4: Train the KNN classifier and measure training time
param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}
knn = KNeighborsClassifier()

results = []  # List to store results

# Manually iterate over parameter grid
for k in param_grid['n_neighbors']:
    start_time = time.time()

    # Set the number of neighbors
    knn.set_params(n_neighbors=k)

    # Fit the model and calculate cross-validation score
    score = GridSearchCV(knn, {'n_neighbors': [k]}, cv=5, scoring='accuracy').fit(train_data_pca,
                                                                                  train_labels).best_score_

    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time

    # Append results to the list
    results.append({'k': k, 'accuracy': score, 'training_time': training_time})

# Step 5: Print results
print("Results:")
for result in results:
    print(f"K={result['k']}, Accuracy={result['accuracy']:.4f}, Training Time={result['training_time']:.2f} seconds")

# Visualize the grid search results
plt.figure(figsize=(10, 6))
plt.plot(param_grid['n_neighbors'], [result['accuracy'] for result in results], marker='o', label='Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Cross-Validated Accuracy')
plt.title('Grid Search Results for KNN')
plt.grid(True)
plt.legend()
plt.show()

# Step 6: Evaluate the best model on the test set
best_k = param_grid['n_neighbors'][np.argmax(mean_scores)]
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(train_data_pca, train_labels)
test_predictions = best_knn.predict(test_data_pca)
accuracy = accuracy_score(test_labels, test_predictions)
print(f"Best KNN Classifier Accuracy: {accuracy * 100:.2f}% with n_neighbors={best_k}")
