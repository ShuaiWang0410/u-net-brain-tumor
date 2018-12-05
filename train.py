#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time, model

from datetime import datetime
#root_path = "/Volumes/PowerExtension"
root_path = "/home/ec2-user"

source_path = "dataset/Processed"
train_path = "train"
test_path = "test"
log_dir = "sw-unet/logs"
model_dir = "sw-unet/models"

test_file_paths = []
train_file_paths = []

image_size = 0
cur_start = 0
train_size, test_size = 0,0
train_images, train_labels, test_images, test_labels = '','','',''

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



def load_data(type = "train"):

    global image_size, train_size, test_size

    t = 0
    if (type == "train"):
        set = train_file_paths
    else:
        set = test_file_paths
        t = 1

    images = np.ndarray(shape=(0, image_size, image_size))
    labels = np.ndarray(shape=(0, image_size, image_size))


    for path in set:
        n = np.load(path)
        images = np.concatenate((images, n[:,0,:,:]))
        labels = np.concatenate((labels, n[:,1,:,:]))


        # max： around 500
        # min:  around 1


        max = 0.
        min = 8.
        for j in range(images.shape[0]):
            if images[j].max() > max:
                max = images[j].max()
            if images[j].min() < min:
                min = images[j].min()

        print(images[0])
        print("Reading " + type + " dataset:" + path + " finished")
    images /= 1.
    images -= 200.

    if t == 0:
        train_size = labels.shape[0]
    else:
        test_size = labels.shape[0]
    labels = labels.astype(np.int32)
    return images, labels

def get_validation(val_size):

    global test_images, test_labels

    val_images = test_images[0:val_size,:,:]
    val_labels = test_labels[0:val_size,:,:]
    x = val_images.shape
    val_images.shape = (x[0], x[1], x[2], 1)

    return val_images, val_labels



def next_batch(batch_size):
    global train_images, train_labels, cur_start

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

def main(task='all'):
    ## Create folder to save trained model and result images

    global train_images, train_labels, test_images, test_labels

    ratios_train = [0,2,4]
    ratios_test = [1]

    split_data(ratios_train,ratios_test)
    train_images, train_labels = load_data("train")
    print("reading training set is done")
    test_images, test_labels = load_data("test")
    print("reading test set is done")
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
    lr = 0.01
    # lr_decay = 0.5
    # decay_every = 100
    beta1 = 0.9
    n_epoch = 3
    print_freq_step = 100
    gpu_frac = 0.99

    n_out = 11
    nw = image_size
    nh = image_size
    nc = 1
    '''
    ###======================== SHOW DATA ===================================###
    # show one slice
    X = np.asarray(X_train[80])
    y = np.asarray(y_train[80])
    # print(X.shape, X.min(), X.max()) # (240, 240, 4) -0.380588 2.62761
    # print(y.shape, y.min(), y.max()) # (240, 240, 1) 0 1
    nw, nh, nz = X.shape
    vis_imgs(X, y, 'samples/{}/_train_im.png'.format(task))
    # show data augumentation results
    for i in range(10):
        x_flair, x_t1, x_t1ce, x_t2, label = distort_imgs([X[:,:,0,np.newaxis], X[:,:,1,np.newaxis],
                X[:,:,2,np.newaxis], X[:,:,3,np.newaxis], y])#[:,:,np.newaxis]])
        # print(x_flair.shape, x_t1.shape, x_t1ce.shape, x_t2.shape, label.shape) # (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1)
        X_dis = np.concatenate((x_flair, x_t1, x_t1ce, x_t2), axis=2)
        # print(X_dis.shape, X_dis.min(), X_dis.max()) # (240, 240, 4) -0.380588233471 2.62376139209
        vis_imgs(X_dis, label, 'samples/{}/_train_im_aug{}.png'.format(task, i))
    '''

    '''===========================GENERATE LOG AND MODEL DIRS ================'''
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    try:
        os.makedirs(os.path.join(model_dir, subdir))
        print("Model directory:" + model_dir)
    except FileExistsError:
        print("Model directory alreadly exist:" + model_dir)

    try:
        os.makedirs(os.path.join(log_dir, subdir))
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

        net_outputs = tl.act.pixel_wise_softmax(net.outputs)
        dice_loss = 1 - tl.cost.dice_coe(net_outputs, t_one_hot_seg)
        iou_loss = tl.cost.iou_coe(net_outputs, t_one_hot_seg)
        dice_hard_loss = tl.cost.dice_hard_coe(net_outputs, t_one_hot_seg)

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


        '''
        ###======================== DEFINE LOSS =========================###
        ## train losses
        out_seg = net.outputs
        dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
        iou_loss = tl.cost.iou_coe(out_seg, t_seg, axis=[0,1,2,3])
        dice_hard = tl.cost.dice_hard_coe(out_seg, t_seg, axis=[0,1,2,3])
        loss = dice_loss

        ## test losses
        test_out_seg = net_test.outputs
        test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
        test_iou_loss = tl.cost.iou_coe(test_out_seg, t_seg, axis=[0,1,2,3])
        test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg, axis=[0,1,2,3])
        '''
        ###======================== DEFINE TRAIN OPTS =======================###
        t_vars = tl.layers.get_variables_with_name('u_net', True, True)
        # with tf.device('/gpu:0'):
        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr, trainable=False)
        train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(dice_loss, var_list=t_vars)

        ###======================== LOAD MODEL ==============================###

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=2)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        tl.layers.initialize_global_variables(sess)
        ## load existing model if possible
        ## tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}.npz'.format(task), network=net)

        ###======================== TRAINING ================================###
        with sess.as_default():
            print("<---------------Start Training the Set---------------->")
            val_images, val_labels = get_validation(10)
            for epoch in range(0, n_epoch):
                epoch_time = time.time()

                # Shuai Wang: training
                print("<----------No." + str(n_epoch + 1) + "epoch started---------->")
                steps = 96 * len(ratios_train)
                for step in range(steps):
                    img_batch, lab_batch = next_batch(batch_size)
                    a, dice_loss_, dice_hard_loss_, iou_loss_ ,_ = sess.run([net_outputs, dice_loss, dice_hard_loss, iou_loss, train_op],
                                                                         feed_dict={t_image:img_batch, t_seg:lab_batch})
                    print("loss for step " + str(step) +  " is " + str(dice_loss_))
                    summary = tf.Summary()
                    summary.value.add(tag='dice_loss', simple_value=dice_loss_)
                    summary.value.add(tag='dice_hard', simple_value=dice_hard_loss_)
                    summary.value.add(tag='iou_loss', simple_value=iou_loss_)
                    summary_writer.add_summary(summary, step)

                    if step % 10 == 0:
                        a, b, x_m = sess.run([net_outputs_val, t_one_hot_seg, accuracy_per_label], feed_dict={t_image:val_images, t_seg:val_labels})
                        print("average accuracy for 11 labels are:")
                        for i in range(11):
                            print("label" + str(i) + ":" + str(x_m[i]))

                print("<----------No." + str(n_epoch + 1) + "epoch ended---------->")


            save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, n_epoch)

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
    '''
    total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
    for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train,
                                batch_size=batch_size, shuffle=True):
        images, labels = batch
        step_time = time.time()
        ## data augumentation for a batch of Flair, T1, T1c, T2 images
        # and label maps synchronously.
        data = tl.prepro.threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis],
                images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis],
                images[:,:,:,3, np.newaxis], labels)],
                fn=distort_imgs) # (10, 5, 240, 240, 1)
        b_images = data[:,0:4,:,:,:]  # (10, 4, 240, 240, 1)
        b_labels = data[:,4,:,:,:]
        b_images = b_images.transpose((0,2,3,1,4))
        b_images.shape = (batch_size, nw, nh, nz)

        ## update network
        _, _dice, _iou, _diceh, out = sess.run([train_op,
                dice_loss, iou_loss, dice_hard, net.outputs],
                {t_image: b_images, t_seg: b_labels})
        total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
        n_batch += 1

        ## you can show the predition here:
        # vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_tmp.png".format(task))
        # exit()

        # if _dice == 1: # DEBUG
        #     print("DEBUG")
        #     vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_debug.png".format(task))

        if n_batch % print_freq_step == 0:
            print("Epoch %d step %d 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)"
            % (epoch, n_batch, _dice, _diceh, _iou, time.time()-step_time))

        ## check model fail
        if np.isnan(_dice):
            exit(" ** NaN loss found during training, stop training")
        if np.isnan(out).any():
            exit(" ** NaN found in output images during training, stop training")

    print(" ** Epoch [%d/%d] train 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)" %
            (epoch, n_epoch, total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch, time.time()-epoch_time))

    ## save a predition of training set
    for i in range(batch_size):
        if np.max(b_images[i]) > 0:
            vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))
            break
        elif i == batch_size-1:
            vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))
    '''
    ###======================== EVALUATION ==========================###
    '''
    total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
    for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                    batch_size=batch_size, shuffle=True):
        b_images, b_labels = batch
        _dice, _iou, _diceh, out = sess.run([test_dice_loss,
                test_iou_loss, test_dice_hard, net_test.outputs],
                {t_image: b_images, t_seg: b_labels})
        total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
        n_batch += 1

    print(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
            (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))
    print(" task: {}".format(task))
    ## save a predition of test set
    for i in range(batch_size):
        if np.max(b_images[i]) > 0:
            vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))
            break
        elif i == batch_size-1:
            vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))

    ###======================== SAVE MODEL ==========================###
    tl.files.save_npz(net.all_params, name=save_dir+'/u_net_{}.npz'.format(task), sess=sess)
    '''

if __name__ == "__main__":
    #import argparse
    #parser = argparse.ArgumentParser()

    #parser.add_argument('--task', type=str, default='all', help='all, necrotic, edema, enhance')

    #args = parser.parse_args()

    #main(args.task)

    image_size = 240

    generate_route()
    main()
