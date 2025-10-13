from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

#MODEL_PATH = "best_pneumonia_model.keras"
#MODEL_PATH = "best_pneumonia_resnet.keras"
MODEL_PATH = "best_pneumonia_resnet_finetuned.keras"
DATA_DIR = r"C:\Users\User\Desktop\archive\chest_xray_split"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def load_trained_model(model_path=MODEL_PATH):
    model = load_model(model_path)
    print(f" Model successfully loaded from: {model_path}")
    return model

def prepare_test_data():
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    return test_gen

def evaluate_model(model, test_gen):
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\n Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    y_pred = model.predict(test_gen)
    y_pred_classes = (y_pred > 0.5).astype("int32").flatten()
    y_true = test_gen.classes

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=list(test_gen.class_indices.keys())))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_gen.class_indices.keys(),
                yticklabels=test_gen.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC Curve)')
    plt.legend(loc="lower right")
    plt.show()

    print(f" AUC Score: {roc_auc:.3f}")

if __name__ == "__main__":
    model = load_trained_model(MODEL_PATH)
    test_gen = prepare_test_data()
    evaluate_model(model, test_gen)
