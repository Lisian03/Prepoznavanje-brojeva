import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target.astype(int)

# Preprocess data
X = X / 255.0
X = X.values.reshape(-1, 28, 28, 1)  # Reshape for CNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(),
}

# Train and evaluate classifiers
results = {}

# Training smaller subset of data for testing speed
X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Prepare the list for plotting
fpr_all = {}
tpr_all = {}
roc_auc_all = {}

for name, clf in classifiers.items():
    print(f"Training {name}...")
    if name in ["Random Forest", "SVM", "KNN","LDA"]:  # These require 2D input
        # Reshape data to 2D for classifiers that do not support 4D input
        X_train_clf = X_train_small.reshape(-1, 28 * 28)
        X_test_clf = X_test.reshape(-1, 28 * 28)
    else:  # CNN handles the 4D data
        X_train_clf = X_train_small
        X_test_clf = X_test

    try:
        clf.fit(X_train_clf, y_train_small)
        y_pred = clf.predict(X_test_clf)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        # For ROC curve, ensure that classifiers support 'predict_proba' method
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test_clf)
            # Calculate ROC curve for the entire set (not per class)
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            fpr_all[name] = fpr
            tpr_all[name] = tpr
            roc_auc_all[name] = roc_auc
        elif hasattr(clf, "decision_function"):  # For classifiers that use decision_function
            y_score = clf.decision_function(X_test_clf)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            fpr_all[name] = fpr
            tpr_all[name] = tpr
            roc_auc_all[name] = roc_auc
        else:
            roc_auc = None

        # Store results
        results[name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Confusion Matrix": cm,
            "ROC AUC": roc_auc,
        }

    except Exception as e:
        print(f"Error training {name}: {e}")

# Build and train a smaller CNN (faster to train)
print("Training smaller CNN...")
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Fewer neurons in the Dense layer

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_small, y_train_small, epochs=2, batch_size=64, verbose=1)

# Evaluate CNN
cnn_pred = model.predict(X_test)
cnn_pred_prob = cnn_pred  # CNN output is already in probability form
cnn_accuracy = accuracy_score(y_test, np.argmax(cnn_pred_prob, axis=1))
cnn_f1 = f1_score(y_test, np.argmax(cnn_pred_prob, axis=1), average='weighted')
cnn_cm = confusion_matrix(y_test, np.argmax(cnn_pred_prob, axis=1))

# Calculate ROC for CNN
fpr, tpr, _ = roc_curve(y_test, cnn_pred_prob[:, 1], pos_label=1)  # Use softmax output for ROC
roc_auc = auc(fpr, tpr)
fpr_all["CNN"] = fpr
tpr_all["CNN"] = tpr
roc_auc_all["CNN"] = roc_auc

# Store CNN results
results["CNN"] = {
    "Accuracy": cnn_accuracy,
    "F1 Score": cnn_f1,
    "Confusion Matrix": cnn_cm,
    "ROC AUC": roc_auc
}

# Build and train a VGG-like CNN
print("Training VGG-like CNN...")
vgg_model = Sequential()
vgg_model.add(Input(shape=(28, 28, 1)))

# VGG-like block 1
vgg_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
vgg_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
vgg_model.add(MaxPooling2D(pool_size=(2, 2)))

# VGG-like block 2
vgg_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vgg_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vgg_model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layers
vgg_model.add(Flatten())
vgg_model.add(Dense(256, activation='relu'))
vgg_model.add(Dense(10, activation='softmax'))

vgg_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
vgg_model.fit(X_train_small, y_train_small, epochs=2, batch_size=64, verbose=1)

# Evaluate VGG model
vgg_pred = vgg_model.predict(X_test)
vgg_pred_prob = vgg_pred  # Softmax output
vgg_accuracy = accuracy_score(y_test, np.argmax(vgg_pred_prob, axis=1))
vgg_f1 = f1_score(y_test, np.argmax(vgg_pred_prob, axis=1), average='weighted')
vgg_cm = confusion_matrix(y_test, np.argmax(vgg_pred_prob, axis=1))

# Calculate ROC for VGG
fpr, tpr, _ = roc_curve(y_test, vgg_pred_prob[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
fpr_all["VGG"] = fpr
tpr_all["VGG"] = tpr
roc_auc_all["VGG"] = roc_auc

# Store VGG results
results["VGG"] = {
    "Accuracy": vgg_accuracy,
    "F1 Score": vgg_f1,
    "Confusion Matrix": vgg_cm,
    "ROC AUC": roc_auc
}


# Display results for all classifiers
for name, metrics in results.items():
    print(f"{name}:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    print(f"Confusion Matrix:\n{metrics['Confusion Matrix']}\n")

# Plot ROC curves separately for each classifier
for name in fpr_all:
    plt.figure()
    plt.plot(fpr_all[name], tpr_all[name], label=f'{name} (AUC = {roc_auc_all[name]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Line for random classifier (diagonal)
    plt.xlabel('Stopa lažno pozitivnih')
    plt.ylabel('Stopa istinitih pozitivnih')
    plt.legend()
    plt.show()
