import argparse
parser = argparse.ArgumentParser(description="Trains and stores a model.")
parser.add_argument('inputdir', type=str,
                    help="Directory containing two subdirectories with images.")
parser.add_argument('output', type=str, default="model.tfm",
                    help="Where to save the trained model.")
parser.add_argument('-e','--epochs', type=int, default=3,
                    help="Number of training epochs.")
parser.add_argument('-f','--finetuning', type=int, default=3,
                    help="Number of finetuning epochs.")
parser.add_argument('-v','--validation', type=float, default=0.2,
                    help="Fraction of data to use for validation.")
parser.add_argument('-b','--batchsize', type=int, default=32,
                    help="Batch size.")
parser.add_argument('-s','--seed', type=int, default=1337,
                    help="Random number seed for repeatability.")
args = vars(parser.parse_args())

### settings

data_dir = args['inputdir']
output_model_filename = args['output']
split = args['validation']
seed = args['seed']
num_epochs_top = args['epochs']
num_epochs_full = args['finetuning']
batch_size = args['batchsize']

###

import tensorflow as tf

print('loading dataset')
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode='binary',
    image_size=(150, 150),
    shuffle=True,
    seed=seed,
    validation_split=split,
    subset='training'
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode='binary',
    image_size=(150, 150),
    shuffle=True,
    seed=seed,
    validation_split=split,
    subset='validation'
)

print("Example data...")
for data, labels in train_ds.take(1):
    print(data.shape)
    print(labels.shape)
print()
print()

size = (150, 150)
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, size), y))

# augmentation
from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
    #[]
)

base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))
x = inputs #data_augmentation(inputs)  # Apply random data augmentation

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1,activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

print("Training last couple of layers...")
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        tf.keras.metrics.Precision(name="prec"),
        tf.keras.metrics.Recall(name="rec")
        ]
)

model.fit(train_ds, epochs=num_epochs_top, validation_data=val_ds)

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
#model.summary()
print("Training full model...")
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        tf.keras.metrics.Precision(name="prec"),
        tf.keras.metrics.Recall(name="rec")
        ]
)

model.fit(train_ds, epochs=num_epochs_full, validation_data=val_ds)

model.save(output_model_filename)