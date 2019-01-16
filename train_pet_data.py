import tensorflow as tf

import sys

sys.path.append('./..')

from datasets import dataset_factory

from nets import nets_factory

from preprocessing import preprocessing_factory

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'cifar10', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './../tmp/cifar10', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'model_name', 'resnet_v2_50', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 224, 'Train image size')

# tf.app.flags.DEFINE_integer('max_number_of_steps', None,
#                             'The maximum number of training steps.')

# tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', '/media/oanaucs/Data/checkpoints_awp/ckpt_resnet_debug2/',
#     'Path to pretrained model for warm start training. If None, cold start.')

tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', None,
    'Path to pretrained model for warm start training. If None, cold start.')

tf.app.flags.DEFINE_string('checkpoint_dir', '/media/oanaucs/Data/pet_adoption',
    'Directory for saving and restoring checkpoints.')


#######################
# Training Flags #
#######################
tf.app.flags.DEFINE_integer(
    'num_epochs', 10,
    'Maximum number of epochs.')

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')


#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Fine-tuning Flags #
#######################

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)

def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


def main():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    g = tf.Graph()
    profiler = tf.profiler.Profiler(g)
    with g.as_default():
        run_meta = tf.RunMetadata()
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7

        # Create global_step
        global_step = tf.train.create_global_step()

        # ######################
        # # Select the dataset #
        # ######################
        # dataset, num_classes, num_samples = dataset_factory.get_dataset(
        # FLAGS.dataset_name, 
        # FLAGS.dataset_split_name, 
        # FLAGS.dataset_dir)

        features_placeholder = tf.placeholder(features.dtype, features.shape)
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        # [Other transformations on `dataset`...]
        dataset = ...
        iterator = dataset.make_initializable_iterator()

        dataset, num_classes, num_samples = dataset, ...

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            is_training=True)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        train_image_size = FLAGS.train_image_size or network_fn.default_image_size

        dataset = dataset.map(lambda image, label: (image_preprocessing_fn(image, train_image_size, train_image_size), label))
        
        dataset = dataset.repeat(FLAGS.num_epochs).shuffle(True).batch(FLAGS.batch_size)

        #########################
        # Load from the dataset #
        #########################
        # make iterator
        # TODO change iterator type
        iterator = dataset.make_one_shot_iterator()

        [images, labels] = iterator.get_next()

        labels -= FLAGS.labels_offset

        labels = tf.one_hot(
          labels, num_classes - FLAGS.labels_offset)

        tf.summary.image('image', images)

        logits, end_points = network_fn(images)

        if 'AuxLogits' in end_points:
            loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
                labels,
                end_points['AuxLogits'],
                label_smoothing=FLAGS.label_smoothing, weights=0.4,
                scope='aux_loss'))
        loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
          labels, logits, label_smoothing=FLAGS.label_smoothing, weights=1.0))

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        total_loss = tf.losses.get_total_loss()

        summaries.add(tf.summary.scalar('loss/%s' % total_loss.op.name, total_loss))
        
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                          tf.nn.zero_fraction(x)))

        # for variable in tf.contrib.framework.get_model_variables():
        #     summaries.add(tf.summary.histogram(variable.op.name, variable))

        model_variables = tf.contrib.framework.get_model_variables()

        variables_to_train = tf.trainable_variables()

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
          variable_averages = tf.train.ExponentialMovingAverage(
              FLAGS.moving_average_decay, global_step)
        else:
          variable_averages = None

        learning_rate = _configure_learning_rate(num_samples, global_step)
        optimizer = _configure_optimizer(learning_rate)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, 
                global_step = global_step,
                var_list = variables_to_train)


        if FLAGS.moving_average_decay:
            with tf.control_dependencies([train_op]):
                # Update ops executed locally by trainer.
                update_ops.append(variable_averages.apply(model_variables))

        restore_model_op = None

        # if pretrained model is specified, init from checkpoint
        if FLAGS.pretrained_model_ckpt_path:
            if tf.gfile.IsDirectory(FLAGS.pretrained_model_ckpt_path):
                variables_to_restore = [var for var in model_variables if ('logits' not in var.op.name.lower())]

                pretrained_model_ckpt_path = tf.train.latest_checkpoint(FLAGS.pretrained_model_ckpt_path)
                tf.logging.info('Fine-tuning from %s' % pretrained_model_ckpt_path)
                restore_model_op = tf.contrib.framework.assign_from_checkpoint_fn(
                  pretrained_model_ckpt_path,
                  variables_to_restore,
                  ignore_missing_vars=FLAGS.ignore_missing_vars)

        with tf.train.MonitoredTrainingSession(checkpoint_dir = FLAGS.checkpoint_dir,
            config = config) as mon_sess:
            if restore_model_op:
                restore_model_op(mon_sess)
            # while not mon_sess.should_stop():
            for i in range(0, 1):
                loss = mon_sess.run([train_op, total_loss], 
                  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                  run_metadata=run_meta)
                profiler.add_step(tf.train.global_step(mon_sess, global_step), run_meta)

        flops = tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('total flops', flops.total_float_ops)

if __name__ == '__main__':
    main()