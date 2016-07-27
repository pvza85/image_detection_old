#!/usr/bin/env python
import argparse
import numpy as np
import tensorflow as tf
import os.path as osp
import os
import models
import dataset
import datetime

class dev_simple_extractor:
    '''class that extract features using tensorflow and save it to output_path'''




    def extract(self):
        '''
        extract features from especific folder
        this the main part that process happens
        return: features of all images in folder
        '''

        # Get the data specifications for the GoogleNet model
        spec = models.get_data_spec(model_class=models.GoogleNet)

        # Create a placeholder for the input image
        input_node = tf.placeholder(tf.float32,
                                    shape=(None, spec.crop_size, spec.crop_size, spec.channels))

        model = getattr(models, self.model, None)
        # Construct the network
        net = model({'data': input_node})
        #net = models.GoogleNet({'data': input_node})



        with tf.Session() as sess:


            # Load the converted parameters
            print('Loading the model')
            net.load(self.model_data, sess)



            # Perform a forward pass through the network to get the class probabilities
            print('Classifying')
            for f in os.listdir(self.dataset_root):
                if os.path.isdir(self.dataset_root + f):
                    folder = self.dataset_root + f + '/'
                    print folder
                    for ff in os.listdir(folder):
                        if os.path.isdir(folder + ff):
                            subfolder = folder + ff + '/'
                            print '\t{0}'.format(subfolder)
                            image_paths = self.image_reader(subfolder)
                            image_producer = dataset.ImageProducer(image_paths=image_paths, data_spec=spec)
                            # Start the image processing workers
                            coordinator = tf.train.Coordinator()
                            threads = image_producer.start(session=sess, coordinator=coordinator)
                            # Create an image producer (loads and processes images in parallel)


                            # Load the input image
                            print('Loading the images')
                            indices, input_images = image_producer.get(sess)

                            tensor = sess.graph.get_tensor_by_name(self.layer)
                            features = sess.run(tensor, feed_dict={input_node: input_images})
                            out_folder = os.path.dirname(image_paths[0]) + '/'
                            out_folder = out_folder.replace(self.dataset_root, self.output_path)
                            if not os.path.exists(out_folder):
                                os.makedirs(out_folder)
                            output_file = '{0}features_{1}_{2}_{3}.txt'.\
                                format(out_folder, self.model, self.layer.replace('/', '').replace(':', ''),
                                       datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
                            #print features.shape
                            with open(output_file, 'wb') as f:
                               np.savetxt(f, features[:,0,0,:], fmt='%.8g')
                            coordinator.request_stop()
                            coordinator.join(threads, stop_grace_period_secs=2)
                            #return features


    def __init__(self, conf):
        # setting configurations
        self.model_data = conf['model_data']
        self.dataset_root = conf['dataset_root']
        self.output_path = conf['output_path']
        self.model = conf['model']
        self.layer = conf['layer']

    def test(self):
        return self.extract()


    def image_reader(self, path):
        '''read images in the folder and return list of their names'''
        image_list = []
        for f in os.listdir(path):
            if f.endswith('.jpg'):
                image_list.append(path + f)
        return sorted(image_list)