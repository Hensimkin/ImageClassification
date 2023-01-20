import numpy as np
import os

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

#class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
#class_names = ['mountain','street']
class_names = ['mountain','street', 'glacier']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)


def load_data():
    """
        Load the data:
            - 14,034 images to train the network.
            - 3,000 images to evaluate how accurately the network learned to classify images.
    """
    datasets = ['C:/Users/hanig/Desktop/train', 'C:/Users/hanig/Desktop/test']
    #datasets = ['C:/Users/hanig/Desktop/seg_train', 'C:/Users/hanig/Desktop/seg_test']
    output = []

    # Iterate through training and test sets
    for dataset in datasets:

        images = []
        labels = []

        print("Loading {}".format(dataset))

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output


(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
n_train = train_labels.shape[0]
n_test = test_labels.shape[0]


print ("Number of training examples: {}".format(n_train))
print ("Number of testing examples: {}".format(n_test))
print ("Each image is of size: {}".format(IMAGE_SIZE))


train_images = train_images / 255.0
test_images = test_images / 255.0


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding = "same", activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), padding = "same", activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding = "same"),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding = "same"),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding = "same"),
    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding = "same"),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 128, activation = tf.nn.relu),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Dense(units = 3, activation = tf.nn.softmax)
])


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=128, epochs=10, validation_split = 0.2)


def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'b-', label = "accuracy")
    plt.plot(history.history['val_accuracy'], 'r-', label = "val_accuracy")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'b-', label = "loss")
    plt.plot(history.history['val_loss'], 'r-', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()


plot_accuracy_loss(history)


test_loss = model.evaluate(test_images, test_labels)


predictions = model.predict(test_images)     # Vector of probabilities
pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability


CM = confusion_matrix(test_labels, pred_labels)
ax = plt.axes()
sn.heatmap(CM, annot=True,
           annot_kws={"size": 10},
           xticklabels=class_names,
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
plt.show()

test_predictions = model.predict(test_images)
test_predictions = np.argmax(test_predictions, axis=1)
accuracy = accuracy_score(test_labels, test_predictions)
print('Accuracy: {:.2f}%'.format(accuracy*100))