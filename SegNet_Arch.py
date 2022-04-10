import tensorflow as tf
from tensorflow import keras
from keras import layers
from Pooling_layers import MaxPooling2DMod, UpSampling2DwithArgmax, MaxUnpooling2D22

def segnet_architecture(input_shape, batch_size=None, num_class=1, 
    version='classic', only_encoder=False, ups=False, freeze=True, padding_vgg16="SAME"):

  def con_batch_relu(inp_ten, out_channels, kernel_size=3, padding='same', relu_act=True):
    # conv lay + batch norm. + relu (act. func.)
    x = layers.Conv2D(out_channels, 
                      kernel_size, 
                      strides=1, 
                      padding=padding)(inp_ten)
    x = layers.BatchNormalization()(x)
    if (relu_act == True):
      x = tf.nn.relu(x)

    return x

  def encoder_block(inp_ten, out_channels):
    number_of_block = len(out_channels)
    x = inp_ten

    for i in range(number_of_block):
      x = con_batch_relu(x, 
                        out_channels[i])

    return MaxPooling2DMod()(x)

  def decoder_block(inp_ten, argmax, out_shape, out_channels, relu_act=True):
    number_of_block = len(out_channels)

    x = UpSampling2DwithArgmax()([inp_ten, argmax], out_shape)
    # x = MaxUnpooling2D22()([inp_ten, argmax], out_shape)         
         
    for i in range(number_of_block):
      x = con_batch_relu(x, 
                        out_channels[i],
                        relu_act=relu_act)

    return x
  
  def last_layer(inp_ten, out_channels, kernel_size=3, padding='same', activation=None):
      x = layers.Conv2D(out_channels, 
                        kernel_size, 
                        padding=padding, 
                        activation=activation)(inp_ten)

      return x

  def saved_model_encoder_block(model, layers_names, block, padding='SAME'):
    for name in (layers_names):
      block =  model.get_layer(name)(block)
    
    return MaxPooling2DMod(padding=padding)(block)

  input_shape_enc = input_shape
  output_shape_dec = [batch_size] + list(input_shape_enc)

  if (version == 'classic'):
    # encoder
    encoder_input = keras.Input(shape=input_shape_enc, batch_size=batch_size, 
                                name="enc_inp_classic")
    x1, x1_idx = encoder_block(encoder_input, [64, 64])
    x2, x2_idx = encoder_block(x1, [128, 128])
    x3, x3_idx = encoder_block(x2, [256, 256, 256])
    x4, x4_idx = encoder_block(x3, [512, 512, 512])
    x5, x5_idx = encoder_block(x4, [512, 512, 512])
    
    if only_encoder:
      # encoder output
      model = keras.Model(encoder_input, x5, name="encoder")
    else:
      # decoder
      x6 = decoder_block(x5, x5_idx, x4, [512, 512, 512])
      x7 = decoder_block(x6, x4_idx, x3, [512, 512, 256])
      x8 = decoder_block(x7, x3_idx, x2, [256, 256, 128])
      x9 = decoder_block(x8, x2_idx, x1, [128, 64])
      x10 = decoder_block(x9, x1_idx, encoder_input, [64, 32])

      # classif.
      x11 = last_layer(x10, num_class, 3, activation='softmax')
    
      # model- output
      model = keras.Model(inputs=encoder_input, outputs=x11, name="SegNetClassic")

  elif (version == 'basic'):
    # encoder
    encoder_input = keras.Input(shape=input_shape_enc, batch_size=batch_size, 
                                name="enc_inp_basic")
    x1, x1_idx = encoder_block(encoder_input, [64])
    x2, x2_idx = encoder_block(x1, [128])
    x3, x3_idx = encoder_block(x2, [256])
    x4, x4_idx = encoder_block(x3, [512])

    # decoder
    x5 = decoder_block(x4, x4_idx, x3, [256], relu_act=False)
    x6 = decoder_block(x5, x3_idx, x2, [128], relu_act=False)
    x7 = decoder_block(x6, x2_idx, x1, [64], relu_act=False)
    x8 = decoder_block(x7, x1_idx, encoder_input, [32], relu_act=False)

    # classif.
    x9 = last_layer(x8, num_class, 3, activation='softmax')

    # model- output
    model = keras.Model(inputs=encoder_input, outputs=x9, name="SegNetBasic")

  elif (version == 'VGG16'):
    input_shape_enc = (224, 224, 3)
    # load vgg16 model from tensorflow               ***********  need to edit input ****************
    model_vgg16 = keras.applications.vgg16.VGG16(
            include_top=False, weights='imagenet',
            input_shape=input_shape_enc, pooling=None
        )

    # freeze layers from pretrained model
    if freeze : model_vgg16.trainable = False 
    
    #get vgg16 layers name
    layer_names = [layer.name for layer in model_vgg16.layers if not '_pool' in layer.name]
    
    # encoder (pre-trained model with argmax MaxPool2D layer)
    # encoder_input = model_vgg16.layers[0].input
    encoder_input = keras.Input(shape=input_shape_enc, batch_size=batch_size, 
                                name="enc_inp_vgg16")

    # batchnorm layers in inference mode when we unfreeze the base model for fine-tuning
    # vgg16 does not have batch normalization layers
    # x = model_vgg16(encoder_input, training=False)

    # this is preprocess operation for vgg16
    # x = keras.applications.mobilenet.preprocess_input(x)

    x1, x1_idx = saved_model_encoder_block(model_vgg16, layer_names[1:3], 
                                          encoder_input, padding=padding_vgg16)       
    x2, x2_idx = saved_model_encoder_block(model_vgg16, layer_names[3:5], x1, 
                                          padding=padding_vgg16)
    x3, x3_idx = saved_model_encoder_block(model_vgg16, layer_names[5:8], x2, 
                                          padding=padding_vgg16)
    x4, x4_idx = saved_model_encoder_block(model_vgg16, layer_names[8:11], x3, 
                                          padding=padding_vgg16)
    x5, x5_idx = saved_model_encoder_block(model_vgg16, layer_names[11:], x4, 
                                          padding=padding_vgg16)

    if only_encoder:
      # encoder output
      model = keras.Model(encoder_input, x5, name="VGG16_encoder")
    else:
      # decoder
      x6 = decoder_block(x5, x5_idx, x4, [512, 512, 512])
      x7 = decoder_block(x6, x4_idx, x3, [512, 512, 256])
      x8 = decoder_block(x7, x3_idx, x2, [256, 256, 128])
      x9 = decoder_block(x8, x2_idx, x1, [128, 64])
      x10 = decoder_block(x9, x1_idx, encoder_input, [64, 32])

      # classif.
      x11 = last_layer(x10, num_class, kernel_size=3, activation='softmax')
    
      # model- output
      model = keras.Model(inputs=encoder_input, outputs=x11, name="SegNetVGG16")

  return model