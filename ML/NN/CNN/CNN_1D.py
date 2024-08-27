####
# This code was written for my MSc project about the binary classification of ECG signals.
#
####

#used for cnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def make_model(depth: int, filters: int, k: int, print_summary = True):
    """
    function to make and compile CNN model

    depth: number of repeats of main block
    filters: number of filters used in each convolutional layer
    k: size of kernel (filter)
    has option to print summary of the model to check succesful compilation

    returns compiled model
    """
    # initialise model
    cnn = Sequential()

    for _ in range(depth):
        cnn.add(Conv1D(filters=filters, kernel_size=k, padding='same', input_shape = (60000, 1)))#input shape should be changed for future use
        cnn.add(BatchNormalization())
        cnn.add(Activation('relu'))
        cnn.add(MaxPooling1D(pool_size=2))  # takes max out of every two
        cnn.add(Dropout(0.5))

    cnn.add(Flatten())
    cnn.add(Dense(1, activation='sigmoid'))  # use 'sigmoid' for binary classification, 'softmax' for multi, also change 1 to number of classes

    # compile model
    cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy']) #binary cross entropy for binary classification, categorical cross entropy for multi

    if print_summary:
        print(cnn.summary())
        
    return cnn


def train_model(model, data: dict, class_weights: dict, EPOCHS: int, BATCH_SIZE: int):
    """
    function to train a compiled model

    model: the compiled model to be trained
    data: dict containg train and test data
    class_weights: dict containing weighting for each class
    EPOCHS: number of epochs of training
    BATCH_SIZE: size of the batches used to perform learning on

    returns the trained model and a dictionary containing the history of the models progress
    """

    #regularisation technique to return the model to the best weights if overfitting occurs
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    history = model.fit(
        data['X_train'], data['y_train'],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(data['X_test'], data['y_test']),
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
    return history, model