import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# hyperparameters
NUM_CLASSES = 2 # TODO - 3 labels
IMG_SHAPE = (160, 576)
BATCH_SZ = 5 # 5
EPOCHS = 40 # 40
LEARN_RATE = 5e-4 # 5e-4
KEEP_PROB = 0.75 # 0.75

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, l3, l4, l7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    k_reg = tf.contrib.layers.l2_regularizer(1e-3)
    k_init = tf.truncated_normal_initializer(stddev = 0.01)

    conv3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                             kernel_regularizer=k_reg, kernel_initializer=k_init)
    conv4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                             kernel_regularizer=k_reg, kernel_initializer=k_init)
    conv7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                             kernel_regularizer=k_reg, kernel_initializer=k_init)

    decv7 = tf.layers.conv2d_transpose(conv7, filters=num_classes, kernel_size=4, strides=(2, 2), padding='same',
                             kernel_regularizer=k_reg, kernel_initializer=k_init)
    skip4 = tf.add(conv4, decv7)

    decv4 = tf.layers.conv2d_transpose(skip4, filters=num_classes, kernel_size=4, strides=(2, 2), padding='same',
                             kernel_regularizer=k_reg, kernel_initializer=k_init)
    skip3 = tf.add(conv3, decv4)

    decv3 = tf.layers.conv2d_transpose(skip3, filters=num_classes, kernel_size=16, strides=(8, 8), padding='same',
                             kernel_regularizer=k_reg, kernel_initializer=k_init)

    return decv3

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits_2d = tf.reshape(nn_last_layer, (-1, num_classes))
    labels_2d = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_2d, logits=logits_2d)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(loss_operation)

    return logits_2d, training_operation, loss_operation

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        ts = time.time()
        print("Processing epoch #{} ...".format(epoch))
        for img, lbl in get_batches_fn(batch_size):
            feed_dict = { input_image: img, correct_label: lbl, keep_prob: KEEP_PROB, learning_rate: LEARN_RATE }
            _, loss = sess.run([ train_op, cross_entropy_loss ], feed_dict=feed_dict)
        ts_delta = time.time() - ts
        print("  Loss={:.4f} dt={}:{}".format(loss, int(ts_delta / 60), int(ts_delta % 60)))


tests.test_train_nn(train_nn)


def run():

    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    learning_rate = tf.placeholder(tf.float32)

    #correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
    #correct_label = tf.placeholder(tf.float32, [None, IMG_SHAPE[0], IMG_SHAPE[1], num_classes])
    correct_label = tf.placeholder(tf.int32, [ None, IMG_SHAPE[0], IMG_SHAPE[1], NUM_CLASSES ], name='correct_label')

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), IMG_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_img, keep_prob, l3, l4, l7 = load_vgg(sess, vgg_path)
        layer_final = layers(l3, l4, l7, NUM_CLASSES)
        logits, training_operation, loss_operation = optimize(layer_final, correct_label, learning_rate=learning_rate, num_classes=NUM_CLASSES)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SZ, get_batches_fn, training_operation, loss_operation, input_img,
             correct_label, keep_prob, learning_rate=learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, IMG_SHAPE, logits, keep_prob, input_img)

        # Save the variables to disk.
        #state_path = os.path.join(data_dir, 'state')
        #save_path = saver.save(sess, "{}/state_{}.ckpt".format(state_path, int(time.time())))
        #print("Model saved in path: %s" % save_path)
        #tf.saved_model.simple_save(sess, export_dir, state_path
        #inputs={"x": x, "y": y},
        #outputs={"z": z})
        #builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
        #builder.add_meta_graph_and_variables(sess, [self._tag])
        #builder.save()

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
