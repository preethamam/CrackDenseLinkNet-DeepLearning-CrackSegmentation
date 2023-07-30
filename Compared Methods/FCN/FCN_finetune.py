from __future__ import print_function, absolute_import, division

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
import os
import scipy.misc as misc
import imageio
from PIL import Image
import tensorflow.contrib.slim as slim

from FCN_DatasetReader import DatasetReader, ImageReader
import FCN_model
import pickle
import torch

WEIGHTS = np.load('vgg19_weights.npy', encoding='bytes', allow_pickle=True).item()

FLAGS = tf.flags.FLAGS
# FCN parameters
tf.flags.DEFINE_string('mode',              'predict',         "Mode of FCN: finetune / predict")
tf.flags.DEFINE_float('learning_rate',      1e-4,               "Learning rate initial value")
tf.flags.DEFINE_float('keep_prob',          0.5,                "Keep probability")
tf.flags.DEFINE_integer('num_of_epoch',     40,                 "Number of epoch")
tf.flags.DEFINE_integer('batch_size',       2,                  "Batch size")

# FCN data parameters
tf.flags.DEFINE_integer('num_of_class',     2,                  "Number of classes")
tf.flags.DEFINE_integer('image_height',     224,                "Heighfinetunet of image")
tf.flags.DEFINE_integer('image_width',      224,                "Width of image")

# FCN storage parameters
tf.flags.DEFINE_string('train_dir',         'data/train',       "Train dataset dir")
tf.flags.DEFINE_string('valid_dir',         'data/valid',       "Valid dataset dir")
delattr(FLAGS, 'log_dir')  # remove 'log_dir' from absl.logging
tf.flags.DEFINE_string('log_dir',           'logs',             "Logs dir")
tf.flags.DEFINE_string('checkpoint_dir',    'checkpoints',      "Checkpoints dir")

# FCN test parameters
tf.flags.DEFINE_string('test_dir',          'test',             "Test dataset dir")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def main(argv=None):

    print(">>> Setting up FCN model ...")

    # model - input
    img_holder, ant_holder = FCN_model.input(FLAGS.image_height, FLAGS.image_width)

    # model - inference
    logits, predictions = FCN_model.inference(img_holder, FLAGS.num_of_class, WEIGHTS, FLAGS.keep_prob)

    # model - loss
    loss_op = FCN_model.loss(logits, ant_holder)
    loss_op_ = FCN_model.loss(logits, ant_holder)

    # model - evaluate
    accuracy = FCN_model.evaluate(predictions, ant_holder)
    precision, recall, f_score, matthews_cc = FCN_model.statistics(predictions, ant_holder)

    # model - train var list
    var_list = tf.trainable_variables()

    # model - train
    train_op = FCN_model.train(loss_op, FLAGS.learning_rate, var_list)

    print(">>> Setting up FCN summary ...")

    # summary - input and predictions
    input_img_sum = tf.summary.image('input_images', img_holder, max_outputs=8)
    input_tru_sum = tf.summary.image('ground_truth', tf.cast(ant_holder * 255, tf.uint8), max_outputs=8)
    input_pre_sum = tf.summary.image('predictions', tf.cast(predictions * 255, tf.uint8), max_outputs=8)

    # summary - train loss
    train_loss = tf.summary.scalar('train_loss', loss_op)
    valid_loss = tf.summary.scalar('valid_loss', loss_op_)

    # summary - merge
    train_summary = tf.summary.merge([input_img_sum, input_tru_sum, input_pre_sum, train_loss])
    valid_summary = tf.summary.merge([valid_loss])

    print(">>> Setting up FCN writer and saver ...")

    # process - summary writer and model saver
    writer = tf.summary.FileWriter(FLAGS.log_dir)
    saver = tf.train.Saver()

    # save train and valid statistics
    train_statistics = []
    valid_statistics = []

    if FLAGS.mode == 'finetune':

        # feed
        train_dataset = DatasetReader(FLAGS.train_dir, [FLAGS.image_height, FLAGS.image_width], True)
        valid_dataset = DatasetReader(FLAGS.valid_dir, [FLAGS.image_height, FLAGS.image_width], False)

        print(">>> Finish loading train dataset and valid dataset ")
       
        #config.gpu_options.allow_growth = True
        #physical_devices = tf.config.experimental.list_physical_devices('GPU')
        #assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        #config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
        #config = tf.config.experimental.set_memory_growth(physical_devices[1], True)
        #config = tf.config.experimental.set_memory_growth(physical_devices[2], True)
        #config = tf.config.experimental.set_memory_growth(physical_devices[3], True)


        with tf.Session(config=config) as sess:
            # initilize model
            init = tf.global_variables_initializer()
            sess.run(init)

            writer.add_graph(sess.graph)

            # if trained, restore the model
            if tf.train.latest_checkpoint(FLAGS.checkpoint_dir):
                print("Load model from {}".format(tf.train.latest_checkpoint(FLAGS.checkpoint_dir)))
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

            # train - parameters
            num_of_epoch = FLAGS.num_of_epoch
            batch_size = FLAGS.batch_size
            num_of_batch = int(train_dataset.num // batch_size)
            num_of_batch_ = int(valid_dataset.num // batch_size)
            step = 0
            # train - main process
            print("============>>>> Begin to train ... <<<<============")
            for epoch in range(num_of_epoch):

                for batch in range(num_of_batch):
                    start_time = time.time()
                    # train batch
                    batch_img, batch_ant = train_dataset.next_batch(batch_size)

                    _, loss, acc, pre, rec, fsc, mcc, summary_str = sess.run([
                        train_op, loss_op, accuracy, precision, recall, f_score, matthews_cc, train_summary],
                        feed_dict={img_holder: batch_img, ant_holder: batch_ant})

                    batch_time = time.time() - start_time

                    # save accuracy and loss
                    train_statistics.append([loss, acc, pre, rec, fsc, mcc, batch_time])

                    print("Epoch: [%d / %d] Batch: [%d / %d] Loss: %.6f, Time: %.3f sec" %
                          (epoch, num_of_epoch, batch, num_of_batch, loss, batch_time))

                    # write train summary
                    writer.add_summary(summary_str, global_step=step)

                    step += 1

                for batch_ in range(num_of_batch_):
                    start_time = time.time()
                    batch_img_, batch_ant_ = valid_dataset.next_batch(batch_size)
                    print(batch_img_.shape, batch_ant_.shape)
                    loss_, acc_, pre_, rec_, fsc_, mcc_, summary_str_ = sess.run([
                        loss_op_, accuracy, precision, recall, f_score, matthews_cc, valid_summary],
                        feed_dict={img_holder: batch_img_, ant_holder: batch_ant_})

                    batch_time_ = time.time() - start_time
                    print("Epoch: [%d / %d] Batch_: [%d / %d] Loss: %.6f, Time: %.3f sec" %
                          (epoch, num_of_epoch, batch_, num_of_batch_, loss_, batch_time_))

                    # save accuracy and loss
                    valid_statistics.append([loss_, acc_, pre_, rec_, fsc_, mcc_, batch_time_])

                    # write valid summary
                    writer.add_summary(summary_str_, global_step=epoch * num_of_batch_ + batch_)

                if (epoch + 1) % 30 == 0:
                    checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'FCN_epoch_' + str(epoch) + '.ckpt')
                    saver.save(sess, checkpoint_file)
                    print("FCN training file {} is saving ... ".format(checkpoint_file))

            print("============>>>> Finish train ... <<<<============")
            save_statistics('train_statistics.npy', train_statistics)
            save_statistics('valid_statistics.npy', valid_statistics)

            print("============>>>> Result save ... <<<<============")

    elif FLAGS.mode == 'predict':
        # images
        image_set = ImageReader(FLAGS.test_dir)

        print(">>> Loading images from test directory ...")

        if tf.train.latest_checkpoint(FLAGS.checkpoint_dir):
            print("Load model from {}".format(tf.train.latest_checkpoint(FLAGS.checkpoint_dir)))
        else:
            print("Please train model first !!!")

        with tf.Session(config=config) as sess:

            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
            
            # Get model summary
            def model_summary():
                model_vars = tf.trainable_variables()
                slim.model_analyzer.analyze_vars(model_vars, print_info=True)

            model_summary()

            # predict - process
            print("============>>>> Begin to predict ... <<<<============")
            print(image_set.num)
            softmax_dict = dict()
            total_time = 0

            for i in range(image_set.num):                

                image, save_name, save_shape = image_set.next_image()

                #resized_image = misc.imresize(image, size=[FLAGS.image_height, FLAGS.image_width])
                resized_image = np.array(Image.fromarray(image).resize((FLAGS.image_width, FLAGS.image_height), Image.NEAREST)).astype(np.double)

                resized_image = np.expand_dims(resized_image, axis=0)

                #prediction = sess.run(predictions, feed_dict={img_holder: resized_image})  # author's code
                
                ##### softmax prediction (used for computing Average Precision) #####
                start_time = time.time()

                prediction = tf.nn.softmax(sess.run(logits, feed_dict={img_holder: resized_image}))
                prediction = 1 - tf.slice(prediction, [0, 0, 0, 0], [1, 224, 224, 1])                
                prediction = tf.reshape(prediction, [224, 224])                
                prediction = prediction.eval(session=tf.Session())
                
                predict_time = time.time() - start_time
                total_time += predict_time


                #####################################################################

                ##### resize and save to dictionary and to pickle #####
                prediction = torch.reshape(torch.from_numpy(prediction), (1, 1, 224, 224))
                m = torch.nn.Upsample(size=save_shape, mode="nearest")  # upsampling
                resized_prediction = m(prediction).numpy()
                softmax_dict[save_name.split("/")[-1]] = list(np.ndarray.flatten(resized_prediction))
                #######################################################
                
                #save_png(save_name, 255*prediction, save_shape)


                print('Image: [%d / %d] Time: %.3f sec' % (i, image_set.num, predict_time))

            print("Average time: " + str(total_time/image_set.num))
            print("============>>>> Finish predict ... <<<<============")
            
            # save dictionary to pickle
            #with open('testing/FCN_' + save_name.split('/')[-3] + '.p', 'wb') as out:
            #	pickle.dump(softmax_dict, out, protocol=pickle.HIGHEST_PROTOCOL)

def save_statistics(file_name, list):
    list_ndarray = np.array(list)
    np.save(file_name, list_ndarray)

def save_png(file_name, ndarray, new_size):
    image = np.squeeze(ndarray)
    #new_image = misc.imresize(image, size=new_size)
    new_image = np.array(Image.fromarray((image).astype(np.uint8)).resize((new_size[1], new_size[0]), Image.NEAREST)).astype(np.double)
    file_name = file_name.split("/")
    file_name[-2] = "pred"
    file_name = "/".join(file_name)
    imageio.imwrite(file_name, new_image)

if __name__ == '__main__':
    tf.app.run()
