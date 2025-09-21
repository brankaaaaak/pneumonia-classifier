from tensorflow.keras.models import load_model

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