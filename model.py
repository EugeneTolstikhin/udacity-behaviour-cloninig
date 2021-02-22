CORRECT_COEFF = 0.2
BATCH_SIZE = 32
EPOCHS_AMOUNT = 3

# Try to use NVIDIA Neural Network for deep learning
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def CNN(train_data=None, valid_data=None, train_step=0., valid_step=0.):
    if train_data == None or valid_data  == None or train_step == 0 or valid_step == 0:
        return

    from keras.layers import Dense, Flatten, Lambda, Conv2D, ReLU, MaxPooling2D, Dropout
    from keras.models import Sequential
    from keras.optimizers import Adam

    model = Sequential()

    model.add(Lambda(
        lambda x: x / 127.5 - 1.,
        input_shape = (66, 200, 3)
    ))

    #model.add(Convolution2D(24, 5, 5, strides=(2, 2)))
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2)))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2)))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2)))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(filters=64, kernel_size=5, strides=(1, 1)))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(filters=64, kernel_size=5, strides=(1, 1)))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(ReLU())

    model.add(Dense(100))
    model.add(ReLU())

    model.add(Dense(50))
    model.add(ReLU())

    model.add(Dense(10))
    model.add(ReLU())
    model.add(Dropout(.5))

    model.add(Dense(1))

    model.compile(optimizer=Adam(.0001), loss="mse")
    model.fit_generator(train_data,
                        validation_data = valid_data,
                        steps_per_epoch = train_step,
                        validation_steps = valid_step,
                        epochs = EPOCHS_AMOUNT,
                        verbose = 1)

    model.save('./model.h5')

def load_images(track_number=1):
    import pandas as pd
    import numpy as np

    col_names = ['center', 'right', 'left', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv('./driving_log_{}.csv'.format(track_number), names=col_names)
    center_imgs = np.array(data.center.tolist())
    left_imgs = np.array(data.left.tolist())
    right_imgs = np.array(data.right.tolist())
    steering = np.array(data.steering.tolist())

    center_imgs = [x[x.rfind('\\') + 1:] for x in center_imgs]
    left_imgs = [x[x.rfind('\\') + 1:] for x in left_imgs]
    right_imgs = [x[x.rfind('\\') + 1:] for x in right_imgs]

    return center_imgs, left_imgs, right_imgs, steering

def generate_training_data(center_imgs_paths, right_imgs_paths, left_imgs_paths, steering):
    import cv2
    import numpy as np
    from sklearn.utils import shuffle

    num_samples = len(center_imgs_paths)
    folder = './IMG/'
    center_imgs = np.empty([])
    right_imgs = np.empty([])
    left_imgs = np.empty([])
    angles = np.empty([])

    while True:
        for offset in range(0, num_samples, BATCH_SIZE):
            imgs = [cv2.imread(folder + path) for path in center_imgs_paths]
            np.append(center_imgs, np.fliplr(imgs))
            angles = np.append(angles, steering)

            imgs = [cv2.imread(folder + path) for path in right_imgs_paths]
            np.append(right_imgs, np.fliplr(imgs))
            np.append(angles, steering + CORRECT_COEFF)

            imgs = [cv2.imread(folder + path) for path in left_imgs_paths]
            np.append(left_imgs, np.fliplr(imgs))
            np.append(angles, steering - CORRECT_COEFF)

            X_train = center_imgs + right_imgs + left_imgs
            y_train = angles

            yield shuffle(X_train, y_train)


from sklearn.model_selection import train_test_split
import math

if __name__ == '__main__':
    center_imgs_paths, left_imgs_paths, right_imgs_paths, steering = load_images(1)

    center_train_paths, center_validation_paths = train_test_split(center_imgs_paths, test_size=0.2)
    right_train_paths, right_validation_paths = train_test_split(right_imgs_paths, test_size=0.2)
    left_train_paths, left_validation_paths = train_test_split(left_imgs_paths, test_size=0.2)
    steering_train, steering_validation = train_test_split(steering, test_size=0.2)

    print(len(center_train_paths))
    print(len(right_train_paths))
    print(len(left_train_paths))
    print(len(steering_train))

    train_data = generate_training_data(center_train_paths, left_train_paths, right_train_paths, steering_train)
    validate_data = generate_training_data(center_validation_paths, left_validation_paths, right_validation_paths, steering_validation)
    
    CNN(train_data, validate_data, math.ceil(len(center_train_paths) / BATCH_SIZE), math.ceil(len(center_validation_paths) / BATCH_SIZE))
    
