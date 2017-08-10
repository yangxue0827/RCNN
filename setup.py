import os
import shutil
import config


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


if __name__ == '__main__':
    # save fine-tune data
    mkdir(config.FINE_TUNE_DATA)
    # save pre-train model
    mkdir(config.SAVE_MODEL_PATH.strip().rsplit('/', 1)[0])
    # save fine-tune model
    mkdir(config.FINE_TUNE_MODEL_PATH.strip().rsplit('/', 1)[0])
    # save svm model and data
    mkdir(os.path.join(config.TRAIN_SVM, '1'))
    mkdir(os.path.join(config.TRAIN_SVM, '2'))