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
    init_dir=None,
    error_plotter=None,
    calculate_train_metrics=False,
    device='cpu'
):
    print(init_dir)
    if 'MDNHead' in str(type(model)):
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

        if error_plotter:
            print('Error plotter not implemented in MDN Head')

        if calculate_train_metrics:
            print('Train metrics not implemented in MDN Head')

    else:

        if error_plotter or calculate_train_metrics:
            val_loss, train_time = train_model_w_metrics(
                prediction_mode,
                model,
                label_encoder,
                train_generator,
                val_generator,
                learning_params,
                save_dir,
                error_plotter,
                calculate_train_metrics,
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
