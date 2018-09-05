chars = [c for c in "abcdefghijklmnopqrstuvwxyz0123456789 "] # includes ctc blank at the last 
#chars = [c for c in "0123456789 "] # includes ctc blank at the last 
num_of_classes = len(chars) + 1 # +1 for blank space = 38
word_max_length = 23 # in Synth 90k dataset, the longest word has 23 letters.
#word_max_length = 5 

blank_label = len(chars)

#img_w, img_h = 256, 128 #128, 32
img_w, img_h = 128, 32

downscalling_factor = 4

batch_size = 256 #100

epochs = 50

train_annotation_filepath = "../raid/mnt/ramdisk/max/90kDICT32px/sample_train.txt"
valid_annotation_filepath = "../data/evaluation_data/svt/test.txt"
test_annotation_filepath = "../data/evaluation_data/icdar13/test.txt"

# train_annotation_filepath = "../data/to_be_labelled_2/train.txt"
# valid_annotation_filepath = "../data/to_be_labelled_2/valid.txt"
# test_annotation_filepath = "../data/bike_handler_crops/annotations.txt"
