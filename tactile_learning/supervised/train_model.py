from tactile_learning.supervised.simple_train_model import simple_train_model
from tactile_learning.supervised.train_model_w_metrics import train_model_w_metrics
from tactile_learning.supervised.train_mdn_model import train_mdn_model

def train_model(
    prediction_mode,
    model,
    label_encoder,
    train_generator,
    val_generator,
    learning_params,
    save_dir,
    device='cpu'
):

    if model.__class__.__name__ in ['MDN_JL', 'MDN_AC']:
        val_loss, train_time = train_mdn_model(
            prediction_mode,
            model,
            label_encoder,
            train_generator,
            val_generator,
            learning_params,
            save_dir,
            device=device
        )

    else:
        val_loss, train_time = simple_train_model(
            prediction_mode,
            model,
            label_encoder,
            train_generator,
            val_generator,
            learning_params,
            save_dir,
            device=device
        )

    return val_loss, train_time
