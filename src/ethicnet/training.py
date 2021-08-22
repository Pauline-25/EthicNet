import tensorflow as tf
import keras
import efficientnet.keras as efn 
from keras import layers

def build_model(nb_classes, image_size, learning_rate = 1e-3 , loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True) , img_augmentation = lambda x : x):
  '''
  Build our model. 
  We take EfficientNetB0 which is not trainible. We add regularized layers in order to make a prediction.

  nb_classes : (required) the number of label classes
  image_size : (required) a tuple a,b with the height and width of the image
  learning rate : (optionnal) default 1e-3
  loss : (optionnal) the loss of the model. Default Categorical Crossentropy
  img_augmentation : (optionnal) the image augmentation to use
  '''

  inputs = tf.keras.layers.Input( shape=image_size+(3,) )
  augmented_inputs = img_augmentation(inputs)
  efficientnet = efn.EfficientNetB0(include_top=False, weights="imagenet",drop_connect_rate=0.2)
  x = efficientnet(augmented_inputs)
  # Freeze the pretrained weights
  efficientnet.trainable = False

  # Rebuild top
  x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
  x = layers.BatchNormalization()(x)
  top_dropout_rate = 0.2
  x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
  if nb_classes == 1 :
      outputs = layers.Dense(nb_classes, activation="sigmoid", name="pred")(x)
  else :
      outputs = layers.Dense(nb_classes, activation="softmax", name="pred")(x)

  # Compile
  model = tf.keras.Model(inputs, outputs)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(
      optimizer=optimizer, loss=loss, metrics=["accuracy"]
  )
  return model

def unfreeze_model(model, nb_layers = 20 , lr=1e-4):
    '''Returns the model, after having unfroze the top nb_layers layers while leaving BatchNorm layers frozen'''
    for layer in model.layers[-nb_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )