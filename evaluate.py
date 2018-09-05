from __future__ import division
from model import get_crnn_model
from keras.optimizers import Adadelta
from image_data_provider import ImageDataProvider, labels_to_text
import params as prms
import numpy as np
import os, argparse, itertools

os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('-w', dest='weight_file', required=True)
args = parser.parse_args()


def decode_label(out):
    # out : (32, 38)
    out = np.expand_dims(out, axis=0)
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(prms.chars):
            outstr += prms.chars[i]
    return outstr

test_data_provider = ImageDataProvider(batch_size=prms.batch_size, annotation_file_path=prms.test_annotation_filepath, img_width=prms.img_w, img_heigth=prms.img_h)
test_data_size = test_data_provider.total_num_imgs
test_steps = test_data_size // prms.batch_size

model = get_crnn_model(training=False)

if args.weight_file and os.path.exists(args.weight_file):
    model.load_weights(args.weight_file)

test_gen = test_data_provider.next_batch()
correct_count = 0
incorrect_count = 0

for _ in range(test_steps):
    next_test_batch = next(test_gen)
    predictions = model.predict(next_test_batch[0]["the_input"])
    labels = next_test_batch[0]["the_labels"]
    for k in range(prms.batch_size):
        prediction = predictions[k]
        label = labels[k]
        
        predicted_label = decode_label(prediction).strip()        
        real_label = labels_to_text(label).strip()
        
        if predicted_label == real_label:            
            correct_count += 1
        else:
            incorrect_count += 1
            print(real_label, predicted_label)
    
print("Accuracy: {} ".format(correct_count / test_data_provider.total_num_imgs))
print("correct count: {}, incorrect_count: {}, all: {}".format(correct_count, incorrect_count, correct_count + incorrect_count))

        



