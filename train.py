from model import get_crnn_model
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator
from image_data_provider import ImageDataProvider
import params as prms
import os, argparse

os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('-w', dest='weight_file')
args = parser.parse_args()

datagen = ImageDataGenerator(
        rotation_range = 8,
        shear_range = 0.15,
        fill_mode = 'nearest',
        zoom_range = 0.15
    )

class LossHistory(Callback):
    def on_train_begin(self, logs={}):        
        self.log_file = open("training_log.txt", 'a')
    
    def on_batch_end(self, batch, logs={}):        
        self.log_file.write('on batch: {}, loss: {}\n'.format(batch, logs.get('loss')))
        # if (batch % 100 == 0):
        self.log_file.flush()
        

train_data_provider = ImageDataProvider(batch_size=prms.batch_size, annotation_file_path=prms.train_annotation_filepath, img_width=prms.img_w, img_heigth=prms.img_h, image_data_generator=datagen, is_augment=True)
train_data_size = train_data_provider.total_num_imgs

valid_data_provider = ImageDataProvider(batch_size=prms.batch_size, annotation_file_path=prms.valid_annotation_filepath, img_width=prms.img_w, img_heigth=prms.img_h, image_data_generator=datagen, is_augment=True)
valid_data_size = valid_data_provider.total_num_imgs

train_steps_per_epoch = train_data_size // prms.batch_size
valid_steps_per_epoch = valid_data_size // prms.batch_size

model = get_crnn_model(training=True)

if args.weight_file and os.path.exists(args.weight_file):
    model.load_weights(args.weight_file)
    
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adadelta())

checkpoint = ModelCheckpoint(filepath='keras_crnn--{epoch:02d}--{val_loss:.3f}.hdf5', verbose=1, save_best_only=True)
loss_logger = LossHistory()

model.fit_generator(generator=train_data_provider.next_batch(), steps_per_epoch=train_steps_per_epoch, epochs=prms.epochs, 
    validation_data=valid_data_provider.next_batch(), validation_steps=valid_steps_per_epoch, callbacks=[checkpoint, loss_logger])
