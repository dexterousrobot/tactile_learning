from tactile_learning.supervised.simple_train_model import simple_train_model


def train_model(
    model,
    label_encoder,
    train_generator,
    val_generator,
    learning_params,
    save_dir,
    device='cpu'
):

    val_loss, train_time = simple_train_model(
        model,
        label_encoder,
        train_generator,
        val_generator,
        learning_params,
        save_dir,
        device=device
    )

    return val_loss, train_time
