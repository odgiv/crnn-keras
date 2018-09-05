from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization, Bidirectional
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
import params as prms


def ctc_lambda(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN 
    # tend to be garbage:
    y_pred = y_pred[:,2:,:]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_crnn_model(training):
    inputs_shape = (prms.img_h, prms.img_w, 1)
    inputs = Input(name="the_input", shape=inputs_shape) #  (None, 32, 128, 1)
    x = Conv2D(64, (3,3), padding="same", name="conv1", activation="relu", kernel_initializer="he_normal")(inputs) # (None, 32, 128, 64)
    x = MaxPooling2D(pool_size=(2,2), name="maxpool1")(x) # (None, 16, 64, 64)

    x = Conv2D(128, (3,3), padding="same", name="conv2", activation="relu", kernel_initializer="he_normal")(x) # (None, 16, 64, 128)
    x = MaxPooling2D(pool_size=(2,2), name="maxpool2")(x) # (None, 8, 32, 128)

    x = Conv2D(256, (3,3), padding="same", name="conv3", activation="relu", kernel_initializer="he_normal")(x) # (None, 8, 32, 256)
    x = Conv2D(256, (3,3), padding="same", name="conv4", activation="relu", kernel_initializer="he_normal")(x) # (None, 8, 32, 256)   
    x = MaxPooling2D(pool_size=(2,1), name="maxpool3")(x) # (None, 4, 32, 256)

    x = Conv2D(512, (3,3), padding="same", name="conv5", kernel_initializer="he_normal")(x) # (None, 4, 32, 512)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)    
    

    x = Conv2D(512, (3,3), padding="same", name="conv6", kernel_initializer="he_normal")(x) # (None, 4, 32, 512)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,1), name="maxpool5")(x) # (None, 2, 32, 512)

    x = Conv2D(512, (2,2), padding="same", name="conv7", activation="relu", kernel_initializer="he_normal")(x) # (None, 2, 32, 512)

    # CRNN to RNN
    
    x = Reshape(target_shape=((32, 1024)), name='reshape')(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(x)

    lstm_1 = LSTM(256, return_sequences=True, kernel_initializer="he_normal", name="lstm1")(x)
    lstm_1b = LSTM(256, return_sequences=True, kernel_initializer="he_normal", name="lstm1_b")(x)
    lstm_1_merged = add([lstm_1, lstm_1b])
    lstm_1_merged = BatchNormalization()(lstm_1_merged)

    lstm_2 = LSTM(256, return_sequences=True, kernel_initializer="he_normal", name="lstm2")(lstm_1_merged)
    lstm_2b = LSTM(256, return_sequences=True, kernel_initializer="he_normal", name="lstm2_b")(lstm_1_merged)
    lstm_2_merged = concatenate([lstm_2, lstm_2b])
    lstm_2_merged = BatchNormalization()(lstm_2_merged)

    x = Dense(prms.num_of_classes, name="dense2")(lstm_2_merged)
    y_pred = Activation('softmax', name="softmax")(x)

    labels = Input(name="the_labels", shape=[prms.word_max_length], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer.

    loss_out = Lambda(ctc_lambda, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])
    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    else:
        return Model(inputs=[inputs], outputs=y_pred)



    

    