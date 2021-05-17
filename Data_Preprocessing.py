# Data Preprocessing
import utils

dest = '../indoorCVPR_09/Images'


# Resizing for VGG16
numpy_array, scenes = utils.resize(224, dest)
utils.dump_into_pkl(numpy_array, "vgg16_images.pkl")
utils.dump_into_pkl(scenes, "scenes.pkl")


# Resizing for Effnetb3
numpy_array, scenes = utils.resize(300, dest)
utils.dump_into_pkl(numpy_array, "Effnetb3_images.pkl")


# Resizing for ResNet50 
numpy_array, scenes = utils.resize(512, dest)
utils.dump_into_pkl(numpy_array, "resnet_images.pkl")
