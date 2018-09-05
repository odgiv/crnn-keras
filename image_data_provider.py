import pandas as pd
import numpy as np
# import cv2
import os.path as path
import params as prms
from skimage.io import imread
from skimage.transform import resize


def text_to_labels(text):
    return list(map(lambda x: prms.chars.index(x), text))

def labels_to_text(label):
    ret = []
    for c in label:
        if c == len(prms.chars):  # CTC Blank
            ret.append("")
        else:
            ret.append(prms.chars[c])
    return "".join(ret)
    

def shuffle_batches(batches):
    # shuffle batch
    random_seed = np.random.random_integers(10)
    for batch in batches:
        np.random.seed(random_seed)
        np.random.shuffle(batch)
    
 
class ImageDataProvider:        

    def __init__(self, batch_size, annotation_file_path, img_width, img_heigth, image_data_generator=None, is_augment=False, num_augment_per_batch=2):
        self.annotation_file_path = annotation_file_path
        self.data_dir = path.dirname(annotation_file_path)        
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_heigth = img_heigth
        self.current_index = 0
        self.image_data_generator = image_data_generator
        self.is_augment = is_augment
        self.num_augment_per_batch = num_augment_per_batch
        

        with open(self.annotation_file_path, "r") as f:      
            self.lines = f.readlines()            
            self.total_num_imgs = len(self.lines)
            np.random.shuffle(self.lines)
            print("read total {} number of images from {}".format(self.total_num_imgs, self.annotation_file_path))
            


    def next_img(self):
        
        while True: 
            if self.total_num_imgs - 1 < self.current_index:
                self.current_index = 0

            line_parts = self.lines[self.current_index].split(' ')
            if len(line_parts) != 2:
                self.current_index += 1
                continue

            filename = line_parts[0].strip()
            label = line_parts[1].strip()
            if (not filename) and label:
                print("No filename for label: {} found.".format(label))                
                self.current_index += 1
                continue

            img_path = path.join(self.data_dir, filename)

            if not path.isfile(img_path):
                print("No image file: {} found.".format(img_path))                
                self.current_index += 1
                continue
            try:
                img = imread(img_path, as_gray=True)
            except (ValueError, OSError, IOError) as err:
                print("Error while reading image: {}, Reason: {} \n".format(img_path, err))
                self.current_index += 1
                continue
            #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)                        

            # If there was somekind of err in an image e.g, premature end of jpeg file.
            # if img is None: 
            #     self.current_index += 1
            #     print("Image that we read is None somehow: {}".format(img_path))
            #     continue

            assert(len(img.shape) == 2)

            # img = cv2.resize(img, (self.img_width, self.img_heigth))
            try:
                img = resize(img, (self.img_heigth, self.img_width), anti_aliasing=True)
            except ValueError as err:
                print("Error while resizing image: {}, Reason: {} \n".format(img_path, err))
                self.current_index += 1
                continue
                
            self.current_index += 1
            yield (img, label)


    def next_batch(self):
        next_img_gen = self.next_img()

        while True:
            x_batch = np.ones((self.batch_size, self.img_heigth, self.img_width, 1), dtype=np.float) 
            y_batch = np.ones((self.batch_size, prms.word_max_length), dtype=np.int) * -1                    
            input_length = np.ones([self.batch_size, 1])
            label_length = np.zeros([self.batch_size, 1])
            
            for i in range(self.batch_size):
                img, text = next(next_img_gen)

                img = np.expand_dims(img, axis=2)

                assert(img.shape == (self.img_heigth, self.img_width, 1))

                x_batch[i] = img
                # For blank labels.
                if text == '':
                    y_batch[i, 0] = prms.blank_label
                    label_length[i] = 1
                else:
                    y_batch[i, 0:len(text)] = text_to_labels(text)
                    label_length[i] = len(text)
                input_length[i] = self.img_width // prms.downscalling_factor - 2 # (128 // 4 - 2) = 30
            

            if not self.is_augment:
                shuffle_batches([x_batch, y_batch, label_length, input_length])

                inputs = {
                    'the_input': x_batch,
                    'the_labels': y_batch,
                    'input_length': input_length,
                    'label_length': label_length
                }
                
                outputs = {
                    'ctc': np.zeros([self.batch_size]) # dummy data for dummy loss function.
                }

                yield (inputs, outputs)
            else:
                # otherwise
                k = 0            
                for x_aug_batch, y_aug_batch in self.image_data_generator.flow(x_batch, y_batch, batch_size=self.batch_size, shuffle=False):                    
                    if k == self.num_augment_per_batch:
                        break
                    k+=1
                    label_length_cp = np.copy(label_length)
                    input_length_cp = np.copy(input_length)
                    
                    shuffle_batches([x_aug_batch, y_aug_batch, label_length_cp, input_length_cp])
                    inputs = {
                        'the_input': x_aug_batch,
                        'the_labels': y_aug_batch,
                        'input_length': input_length_cp,
                        'label_length': label_length_cp
                    }
                    
                    outputs = {
                        'ctc': np.zeros([self.batch_size]) # dummy data for dummy loss function.
                    }

                    yield (inputs, outputs)                       


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from keras.preprocessing.image import ImageDataGenerator

    def plot_images(images, labels, nrows, ncols, cls_true=None, cls_pred=None, grey=False):
        """ Helper function for plotting nrows * ncols images
        """
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2*nrows))

        for i, ax in enumerate(axes.flat):         
            if grey:
                ax.imshow(images[i,:,:,0], cmap='binary')
            else:
                ax.imshow(images[i])

            ax.set_xticks([]); ax.set_yticks([])
            if labels:
                ax.set_title(labels[i])

    datagen = ImageDataGenerator(
        rotation_range = 15, # 8
        shear_range = 0.15, 
        fill_mode = 'nearest',
        zoom_range = 0.15,
        # added later
        width_shift_range = 0.1,
        height_shift_range = 0.1
    )

    # img_data_provider = ImageDataProvider(10, "I:\\sampleSyntImgs\\annotations.txt", 128, 32, image_data_generator=datagen, is_augment=True)
    img_data_provider = ImageDataProvider(10, "H:\\Others\\to_be_labelled_2\\train.txt", 256, 128, image_data_generator=datagen, is_augment=True)

    generator = img_data_provider.next_batch()

    for _ in range(6): # if is_augment is true, it should show 3 different set of images due to 2 augmentations per batch. Otherwise 6 sets of images.

        batch = next(generator)

        imgs = batch[0]["the_input"]
        labels = batch[0]["the_labels"]
        
        text_labels = [labels_to_text(label) for label in labels]

        plot_images(imgs, text_labels, 2, 5, grey=True)

        plt.show()

