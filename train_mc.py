#! /usr/bin/python
# -*- coding: utf8 -*-
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time, model

from datetime import datetime
from PIL import Image
#root_path = "/Volumes/PowerExtension"
root_path = "/home/ec2-user"

# source_path = "dataset/Processed_2"
source_path = "Processed_2"
train_path = "train"
test_path = "test"
log_dir = "sw-unet/logs"
model_dir = "sw-unet/models"
ratios_train = [0,2,3,4,6]
ratios_test = [1,5]
#ratios_train = [0]
#ratios_test = [1]

test_file_paths = []
train_file_paths = []

image_size = 0
cur_start = 0
cur_start_val = 0
train_size, test_size = 0,0
train_images, train_labels, test_images, test_labels = '','','',''
elastic_pairs = [[720,24], [1000, 40], [1000, 60], [1000, 80]]

def elastic_transformations(alpha, sigma, rng=np.random.RandomState(42),
                            interpolation_order=1):
    def _elastic_transform_2D(images):
        """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
        # Take measurements
        image_shape = images[0].shape
        # Make random fields
        dx = rng.uniform(-1, 1, image_shape) * alpha
        dy = rng.uniform(-1, 1, image_shape) * alpha
        # Smooth dx and dy
        sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
        sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
        # Make meshgrid
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        # Distort meshgrid indices
        distorted_indices = (y + sdy).reshape(-1, 1), \
                            (x + sdx).reshape(-1, 1)

        # Map cooordinates from image to distorted index set

        transformed_images = [map_coordinates(image, distorted_indices, mode='reflect',
                                              order=interpolation_order).reshape(image_shape)
                              for image in images]
        return transformed_images
    return _elastic_transform_2D


def distort_imgs(data):
    """ data augumentation """
    x1, x2, x3, x4, y = data
    # x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],  # previous without this, hard-dice=83.7
    #                         axis=0, is_random=True) # up down
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],
                            axis=1, is_random=True) # left right
    x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y],
                            alpha=720, sigma=24, is_random=True)
    x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20,
                            is_random=True, fill_mode='constant') # nearest, constant
    x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10,
                            hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05,
                            is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y],
                            zoom_range=[0.9, 1.1], is_random=True,
                            fill_mode='constant')
    return x1, x2, x3, x4, y

def vis_imgs(X, y, path):
    """ show one slice """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y]), size=(1, 5),
        image_path=path)

def vis_imgs2(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y_, y]), size=(1, 6),
        image_path=path)

# Shuai Wang

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)

def generate_route():

    global root_path, source_path
    global train_path, test_path
    global log_dir, model_dir

    source_path = os.path.join(root_path, source_path)
    train_path = os.path.join(source_path, train_path)
    test_path = os.path.join(source_path, test_path)
    log_dir = os.path.join(root_path, log_dir)
    model_dir = os.path.join(root_path, model_dir)


def split_data(training_series, test_series):

    global test_file_paths, train_file_paths

    for i in training_series:
        train_file_paths.append(os.path.join(source_path, str(i) + ".npy"))

    for i in test_series:
        test_file_paths.append(os.path.join(source_path, str(i) + ".npy"))

def load_data_mc(type = "train"):

    global image_size, train_size, test_size

    t = 0
    if (type == "train"):
        set = train_file_paths
    else:
        set = test_file_paths
        t = 1

    full = np.ndarray(shape=(0, image_size, image_size, 5))

    images = np.ndarray(shape=(0, image_size, image_size, 4))
    labels = np.ndarray(shape=(0, image_size, image_size))

    for path in set:
        n = np.load(path)
        full = np.concatenate((full, n))

        # max： around 500
        # min:  around 1

        max = 0.
        min = 8.
        for j in range(images.shape[0]):
            if images[j].max() > max:
                max = images[j].max()
            if images[j].min() < min:
                min = images[j].min()

        # print(images[0])
        print("Reading " + type + " dataset:" + path + " finished")
    np.random.shuffle(full)
    images = np.concatenate((images, full[:, :, :, 0:4]))
    labels = np.concatenate((labels, full[:, :, :, 4]))
    images /= 100.

    if t == 0:
        train_size = labels.shape[0]
    else:
        test_size = labels.shape[0]
    labels = labels.astype(np.int32)
    return images, labels

def get_inf_mc(number):
    global test_images, test_labels
    img_batch = test_images[number, :, :, :]
    x = img_batch.shape
    img_batch.shape = (1, x[0], x[1], x[2])
    lab_batch = test_labels[number, :, :]
    y = lab_batch.shape
    lab_batch.shape = (1, y[0], y[1])
    return img_batch, lab_batch

def get_validation_mc(val_size):
    '''
    global test_images, test_labels,
    val_images = test_images[0:val_size,:,:]
    val_labels = test_labels[0:val_size,:,:]
    x = val_images.shape
    val_images.shape = (x[0], x[1], x[2], 1)

    return val_images, val_labels
    '''
    global test_images, test_labels, cur_start_val, test_size

    start = cur_start_val
    if start + val_size >= test_size:
        end = train_size
        cur_start_val = 0
    else:
        end = start + val_size
        cur_start_val = end

    img_batch = test_images[start:end, :, :,:]
    lab_batch = test_labels[start:end, :, :]
    return img_batch, lab_batch


def next_batch_mc(batch_size):
    global train_images, train_labels, cur_start, train_size

    start = cur_start
    if start + batch_size >= train_size:
        end = train_size
        cur_start = 0
    else:
        end = start + batch_size
        cur_start = end

    img_batch = train_images[start:end,:,:,:]
    lab_batch = train_labels[start:end,:,:]
    return img_batch, lab_batch



def load_data(type = "train"):

    global image_size, train_size, test_size

    t = 0
    if (type == "train"):
        set = train_file_paths
    else:
        set = test_file_paths
        t = 1

    full = np.ndarray(shape=(0,2,image_size,image_size))

    images = np.ndarray(shape=(0, image_size, image_size))
    labels = np.ndarray(shape=(0, image_size, image_size))


    for path in set:
        n = np.load(path)
        full = np.concatenate((full, n))


        # max： around 500
        # min:  around 1


        max = 0.
        min = 8.
        for j in range(images.shape[0]):
            if images[j].max() > max:
                max = images[j].max()
            if images[j].min() < min:
                min = images[j].min()

        # print(images[0])
        print("Reading " + type + " dataset:" + path + " finished")
    np.random.shuffle(full)
    images = np.concatenate((images, full[:, 0, :, :]))
    labels = np.concatenate((labels, full[:, 1, :, :]))
    images /= 100.
    images -= 20.

    if t == 0:
        train_size = labels.shape[0]
    else:
        test_size = labels.shape[0]
    labels = labels.astype(np.int32)
    return images, labels

def get_inf(number):
    global test_images, test_labels
    img_batch = test_images[number, :, :]
    x = img_batch.shape
    img_batch.shape = (1, x[0], x[1], 1)
    lab_batch = test_labels[number, :, :]
    y = lab_batch.shape
    lab_batch.shape = (1, x[0], x[1])
    return img_batch, lab_batch

def get_validation(val_size):
    '''
    global test_images, test_labels,
    val_images = test_images[0:val_size,:,:]
    val_labels = test_labels[0:val_size,:,:]
    x = val_images.shape
    val_images.shape = (x[0], x[1], x[2], 1)

    return val_images, val_labels
    '''
    global test_images, test_labels, cur_start_val, test_size

    start = cur_start_val
    if start + val_size >= test_size:
        end = train_size
        cur_start_val = 0
    else:
        end = start + val_size
        cur_start_val = end

    img_batch = test_images[start:end, :, :]
    x = img_batch.shape
    img_batch.shape = (x[0], x[1], x[2], 1)
    lab_batch = test_labels[start:end, :, :]
    return img_batch, lab_batch


def next_batch(batch_size):
    global train_images, train_labels, cur_start, train_size

    start = cur_start
    if start + batch_size >= train_size:
        end = train_size
        cur_start = 0
    else:
        end = start + batch_size
        cur_start = end

    img_batch = train_images[start:end,:,:]
    x = img_batch.shape
    img_batch.shape = (x[0], x[1], x[2], 1)
    lab_batch = train_labels[start:end,:,:]
    return img_batch, lab_batch


# Shuai Wang

def main(args):
    ## Create folder to save trained model and result images

    global train_images, train_labels, test_images, test_labels
    global model_dir, log_dir

    split_data(ratios_train,ratios_test)
    train_images, train_labels = load_data_mc("train")
    print("reading training set is done, train size is "+str(train_size/48))
    test_images, test_labels = load_data_mc("test")
    print("reading test set is done, test size is "+str(test_size/48))
    '''
    save_dir = "checkpoint"
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir("samples/{}".format(task))

    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target
    # there are 4 labels in targets:
    # Label 0: background
    # Label 1: necrotic and non-enhancing tumor
    # Label 2: edema
    # Label 4: enhancing tumor
    import prepare_data_with_valid as dataset
    X_train = dataset.X_train_input
    y_train = dataset.X_train_target[:,:,:,np.newaxis]
    X_test = dataset.X_dev_input
    y_test = dataset.X_dev_target[:,:,:,np.newaxis]

    if task == 'all':
        y_train = (y_train > 0).astype(int)
        y_test = (y_test > 0).astype(int)
    elif task == 'necrotic':
        y_train = (y_train == 1).astype(int)
        y_test = (y_test == 1).astype(int)
    elif task == 'edema':
        y_train = (y_train == 2).astype(int)
        y_test = (y_test == 2).astype(int)
    elif task == 'enhance':
        y_train = (y_train == 4).astype(int)
        y_test = (y_test == 4).astype(int)
    else:
        exit("Unknow task %s" % task)
    '''
    ###======================== HYPER-PARAMETERS ============================###
    batch_size = 5
    lr = 0.000001
    # lr_decay = 0.5
    # decay_every = 100
    beta1 = 0.9
    n_epoch = 3
    print_freq_step = 600
    gpu_frac = 0.99

    n_out = 11
    nw = image_size
    nh = image_size
    nc = 4

    '''===========================GENERATE LOG AND MODEL DIRS ================'''
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_dir = os.path.join(model_dir, subdir)
    log_dir = os.path.join(log_dir, subdir)
    try:
        os.makedirs(model_dir)
        print("Model directory:" + model_dir)
    except FileExistsError:
        print("Model directory alreadly exist:" + model_dir)

    try:
        os.makedirs(log_dir)
        print("Log directory alreadly exist:" + log_dir)
    except FileExistsError:
        print("Log directory alreadly exist:" + log_dir)


    '''===========================Network Compile============================='''

    # with tf.device('/cpu:0'):
    with tf.Graph().as_default():

        # with tf.device('/gpu:0'): #<- remove it if you train on CPU or other GPU
        ###======================== DEFIINE MODEL =======================###
        ## nz is 4 as we input all Flair, T1, T1c and T2.
        t_image = tf.placeholder('float32', [None, nw, nh, nc], name='input_image')
        ## labels are either 0 or 1
        ##t_seg = tf.placeholder('float32', [batch_size, nw, nh, n_out], name='target_segment')
        t_seg = tf.placeholder('int32', [None, nw, nh], name='target_segment')

        t_one_hot_seg = tf.keras.backend.one_hot(t_seg, num_classes=11)

        ## train inference
        net = model.u_net(t_image, is_train=True, reuse=False, n_out=n_out)
        ## test inference
        net_val = model.u_net(t_image, is_train=False, reuse=True, n_out=n_out)
        net_test = model.u_net(t_image, is_train=False, reuse=True, n_out=n_out)

        '''======================== DEFINE MY LOSS ======================'''

        net_output = tl.act.pixel_wise_softmax(net.outputs)
        dice_loss = 1 - tl.cost.dice_coe(net_output, t_one_hot_seg, axis=(0,1,2,3))
        iou_loss = tl.cost.iou_coe(net_output, t_one_hot_seg, axis=(0,1,2,3))
        dice_hard_loss = tl.cost.dice_hard_coe(net_output, t_one_hot_seg, axis=(0,1,2,3))

        '''======================== DEFINE ACCURACY ====================='''
        net_outputs_val = tl.act.pixel_wise_softmax(net_val.outputs)
        # net_outputs_val = net_val.outputs
        # tf.reduce_mean(tf.cast(tf.equal(net_outputs_val[:, :, :, 1]3), tf.argmax(t_one_hot_seg, 3)), tf.float32), axis = (1, 2)))
        accuracy_per_label = []
        # shape = t_image.get_shape().as_list()  # a list: [None, 9, 2]
        # image_dim = np.prod(shape[1:])  # dim = prod(9,2) = 18
        net_outputs_val = tf.one_hot(tf.argmax(net_outputs_val, 3), depth=n_out)
        for i in range(0,11):
            # accuracy_per_label.append(tf.reduce_mean(tf.cast(tf.keras.metrics.binary_accuracy(net_outputs_val[:,:,:,i], t_one_hot_seg[:,:,:,i], threshold=0.5), tf.float32), axis=(1,2)))
            #accuracy_per_label.append(tf.keras.metrics.binary_accuracy(tf.reshape(net_outputs_val[:,:,:,i], shape=[-1, image_dim]),
            #                                                           tf.reshape(t_one_hot_seg[:,:,:,i], shape=[-1, image_dim]),
            #                                                           threshold=0.5))
            accuracy_per_label.append(tf.reduce_mean(tf.reduce_mean(tf.cast(tf.equal(net_outputs_val[:, :, :, i], t_one_hot_seg[:,:,:,i]), tf.float32), axis=(1,2))))

        '''======================== DEFINE INFERENCE ======================'''
        net_inf = tf.argmax(net_test.outputs, 3)
        infer = tf.reduce_mean(tf.cast(tf.equal(net_inf, tf.argmax(t_one_hot_seg, 3)), tf.float32), axis = (1,2))

        ###======================== DEFINE TRAIN OPTS =======================###
        t_vars = tl.layers.get_variables_with_name('u_net', True, True)
        # with tf.device('/gpu:0'):
        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr, trainable=False)
        train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(dice_loss, var_list=t_vars)

        ###======================== CONFIG MODEL ==============================###

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=2)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        # if (args.task == 'training'):
        tl.layers.initialize_global_variables(sess)
        ## load existing model if possible
        ## tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}.npz'.format(task), network=net)

        g_step = 0
        sw = 0
        ###======================== TRAINING ================================###
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        with sess.as_default():
            if (args.task == 'training'):
                print("<---------------Start Training the Set---------------->")
                # val_images, val_labels = get_validation(batch_size * 4)
                for epoch in range(1, n_epoch + 1):
                    epoch_time = time.time()

                    # Shuai Wang: training
                    print("<----------No." + str(epoch) + "epoch started---------->")
                    steps = int(((48 * len(ratios_train))/ batch_size) )
                    if (epoch % 2 == 1):
                        sw += 1
                        sw = sw % 4
                    for step in range(steps):
                        img_batch, lab_batch = next_batch_mc(batch_size)

                        if (epoch % 2 == 1):
                            '''
                            data = tl.prepro.threading_data([_ for _ in zip(img_batch[:, :, :, 0, np.newaxis],
                                                                            img_batch[:, :, :, 1, np.newaxis],
                                                                            img_batch[:, :, :, 2, np.newaxis],
                                                                            img_batch[:, :, :, 3, np.newaxis], lab_batch)],
                                                            fn=distort_imgs)  # (10, 5, 240, 240, 1)
                            b_images = data[:, 0:4, :, :, :]  # (10, 4, 240, 240, 1)
                            lab_batch = data[:, 4, :, :, :]
                            img_batch = b_images.transpose((0, 2, 3, 1, 4))
                            img_batch.shape = (batch_size, image_size, image_size, nc)
                            lab_batch.shape = (batch_size, image_size, image_size)
                            
                            x_flair, x_t1, x_t1ce, x_t2, lab_batch = distort_imgs(
                            [img_batch[:, :, :, 0, np.newaxis],
                             img_batch[:, :, :, 1, np.newaxis],
                             img_batch[:, :, :, 2, np.newaxis],
                             img_batch[:, :, :, 3, np.newaxis], lab_batch[:, :, :, np.newaxis]])      
                             
                             img_batch = np.concatenate((x_flair, x_t1, x_t1ce, x_t2), axis = 3)
                            # img_batch = img_batch.transpose((0, 2, 3, 1, 4))
                            # img_batch.shape = (batch_size, image_size, image_size, nc)
                            lab_batch.shape = (batch_size, image_size, image_size)              
                            '''
                            x_s = []
                            for i in range(batch_size):
                                cut = elastic_transformations(elastic_pairs[sw][0], elastic_pairs[sw][1])(
                                                                        [img_batch[i, :, :, 0],
                                                                         img_batch[i, :, :, 1],
                                                                         img_batch[i, :, :, 2],
                                                                         img_batch[i, :, :, 3], lab_batch[i, :, :]])
                                cob = np.ndarray(shape = (240, 240, 0))
                                for j in cut:
                                    j.shape = (image_size, image_size, 1)
                                    cob = np.concatenate((cob, j), axis=2)
                                x_s.append(cob)
                            x_s = np.array(x_s)
                            img_batch, lab_batch = x_s[:,:,:,0:4], x_s[:,:,:,4]

                        a, dice_loss_, dice_hard_loss_, iou_loss_ ,_ = sess.run([net_output, dice_loss, dice_hard_loss, iou_loss, train_op],
                                                                             feed_dict={t_image:img_batch, t_seg:lab_batch})
                        print(str(epoch) + ": loss for step " + str(step) +  " is " + str(dice_loss_))
                        summary = tf.Summary()
                        summary.value.add(tag='dice_loss', simple_value=dice_loss_)
                        summary.value.add(tag='dice_hard', simple_value=dice_hard_loss_)
                        summary.value.add(tag='iou_loss', simple_value=iou_loss_)
                        total_dice += dice_loss_
                        total_iou += iou_loss_
                        total_dice_hard += dice_hard_loss_

                        if step % 6 == 0:
                            val_images, val_labels = get_validation_mc(48)
                            a, b, x_m = sess.run([net_outputs_val, t_one_hot_seg, accuracy_per_label], feed_dict={t_image:val_images, t_seg:val_labels})
                            print("average accuracy for 11 labels are:")
                            # image = tf.image.decode_png(a[2] * 20, channels=1)
                            # image = tf.expand_dims(image, 0)
                            # sess = tf.Session()
                            # writer = tf.summary.FileWriter('logs')
                            # summary.image("image1", image)
                            print("<---------------Start Evaluation" + str(step) + "the Set---------------->")
                            for i in range(11):
                                print("label" + str(i) + ":" + str(x_m[i]))
                                summary.value.add(tag='accuracy_label'+str(i), simple_value=x_m[i])
                            summary.value.add(tag='accuracy_avg', simple_value=np.mean(x_m))
                            print("<---------------End Evaluation" + str(step) + "the Set---------------->")
                        summary_writer.add_summary(summary, g_step)
                        g_step += 1


                    print("<----------No." + str(n_epoch + 1) + "epoch ended---------->")
                summary = tf.Summary()
                summary.value.add(tag='total_dice_loss', simple_value=total_dice)
                summary.value.add(tag='total_dice_hard', simple_value=total_dice_hard)
                summary.value.add(tag='total_iou_loss', simple_value=total_iou)
                summary_writer.add_summary(summary, g_step)

                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, n_epoch)

            if (args.task == 'inference'):
                # saver.restore(sess, args.model)
                r_image = Image.fromarray(test_images[20])
                r_image.show(title="origin image")
                r_label = Image.fromarray(test_labels[20] * 30)
                r_label.show(title="origin label")
                test_image, test_label = get_inf_mc(20)
                net_inf_, infer_ = sess.run([net_inf, infer], feed_dict={t_image:test_image,t_seg:test_label})
                net_inf_.shape = (image_size, image_size)
                net_inf_ = net_inf_.astype(np.float64)
                p_label = Image.fromarray(net_inf_ * 30)
                p_label.show(title="predicted label")


    # Shuai Wang: training end

    ## update decay learning rate at the beginning of a epoch
    # if epoch !=0 and (epoch % decay_every == 0):
    #     new_lr_decay = lr_decay ** (epoch // decay_every)
    #     sess.run(tf.assign(lr_v, lr * new_lr_decay))
    #     log = " ** new learning rate: %f" % (lr * new_lr_decay)
    #     print(log)
    # elif epoch == 0:
    #     sess.run(tf.assign(lr_v, lr))
    #     log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
    #     print(log)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='training', help='training or inference')
    parser.add_argument('--model', type=str,
                        default='/Volumes/PowerExtension/Pretrained_models/sw-unet/models/20181206-042805/model-20181206-042805.ckpt-6',
                        help='set a pre-trained model path')
    image_size = 240

    generate_route()

    args = parser.parse_args()

    main(args)
