from __future__ import division, print_function, absolute_import
import numpy as np
import selectivesearch
import os.path
from sklearn import svm
from sklearn.externals import joblib
import preprocessing_RCNN as prep
import os
import tools
import cv2
import config
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def image_proposal(img_path):
    img = cv2.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(
                       img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding small regions
        if r['size'] < 220:
            continue
        if (r['rect'][2] * r['rect'][3]) < 500:
            continue
        # resize to 227 * 227 for input
        proposal_img, proposal_vertice = prep.clip_pic(img, r['rect'])
        # Delete Empty array
        if len(proposal_img) == 0:
            continue
        # Ignore things contain 0 or not C contiguous array
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        # Check if any 0-dimension exist
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        resized_proposal_img = prep.resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices


# Load training images
def generate_single_svm_train(train_file):
    save_path = train_file.rsplit('.', 1)[0].strip()
    if len(os.listdir(save_path)) == 0:
        print("reading %s's svm dataset" % train_file.split('\\')[-1])
        prep.load_train_proposals(train_file, 2, save_path, threshold=0.3, is_svm=True, save=True)
    print("restoring svm dataset")
    images, labels = prep.load_from_npy(save_path)

    return images, labels


# Use a already trained alexnet with the last layer redesigned
def create_alexnet():
    # Building 'AlexNet'
    network = input_data(shape=[None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


# Construct cascade svms
def train_svms(train_file_folder, model):
    files = os.listdir(train_file_folder)
    svms = []
    for train_file in files:
        if train_file.split('.')[-1] == 'txt':
            X, Y = generate_single_svm_train(os.path.join(train_file_folder, train_file))
            train_features = []
            for ind, i in enumerate(X):
                # extract features
                feats = model.predict([i])
                train_features.append(feats[0])
                tools.view_bar("extract features of %s" % train_file, ind + 1, len(X))
            print(' ')
            print("feature dimension")
            print(np.shape(train_features))
            # SVM training
            clf = svm.LinearSVC()
            print("fit svm")
            clf.fit(train_features, Y)
            svms.append(clf)
            joblib.dump(clf, os.path.join(train_file_folder, str(train_file.split('.')[0]) + '_svm.pkl'))
    return svms


if __name__ == '__main__':
    train_file_folder = config.TRAIN_SVM
    img_path = './17flowers/jpg/7/image_0591.jpg'  # or './17flowers/jpg/16/****.jpg'
    imgs, verts = image_proposal(img_path)
    tools.show_rect(img_path, verts)

    net = create_alexnet()
    model = tflearn.DNN(net)
    model.load(config.FINE_TUNE_MODEL_PATH)
    svms = []
    for file in os.listdir(train_file_folder):
        if file.split('_')[-1] == 'svm.pkl':
            svms.append(joblib.load(os.path.join(train_file_folder, file)))
    if len(svms) == 0:
        svms = train_svms(train_file_folder, model)
    print("Done fitting svms")
    features = model.predict(imgs)
    print("predict image:")
    print(np.shape(features))
    results = []
    results_label = []
    count = 0
    for f in features:
        for svm in svms:
            pred = svm.predict([f.tolist()])
            # not background
            if pred[0] != 0:
                results.append(verts[count])
                results_label.append(pred[0])
        count += 1
    print("result:")
    print(results)
    print("result label:")
    print(results_label)
    tools.show_rect(img_path, results)










