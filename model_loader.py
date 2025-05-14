import tensorflow as tf
from tensorflow.keras import backend as K
from keras.layers import Conv2DTranspose  # Updated import

# Custom metrics with __name__ attributes
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
dice_coef.__name__ = "dice_coef"  # Critical fix

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
dice_coef_loss.__name__ = "dice_coef_loss"  # Critical fix

# Custom Conv2DTranspose layer (fix groups parameter)
class FixedConv2DTranspose(Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)  # Remove 'groups' if present
        super().__init__(*args, **kwargs)

def load_unet_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_coef": dice_coef,
            "dice_coef_loss": dice_coef_loss,
            "Conv2DTranspose": FixedConv2DTranspose
        }
    )