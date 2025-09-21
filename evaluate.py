from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = r'C:\Users\User\Desktop\archive\chest_xray_split'

def load_trained_model(model_path="best_pneumonia_model.h5"):
    """
    Loads a saved Keras model from a .h5 file.
    :param model_path: path to the model file
    :return: loaded model ready for evaluation or predictions
        """
    model = load_model(model_path)
    print(f"The model was loaded from {model_path}")
    return model

if __name__ == "__main__":
    model = load_trained_model("best_pneumonia_model.h5")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    # evaluation
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"📊 Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    # predictions on the test set 
    y_pred = model.predict(test_gen)
    y_pred_classes = (y_pred > 0.5).astype("int32").flatten()
    y_true = test_gen.classes

    # classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=list(test_gen.class_indices.keys())))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_gen.class_indices.keys(),
                yticklabels=test_gen.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()