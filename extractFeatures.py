from os import listdir
from pickle import dump,load
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

#extract features from each photo in the directory
def extract_features(directory):
    
    #load the model
    model = VGG16()
    
    #print(model.summary())
    #restructure the model
    model.layers.pop()
    model = Model(inputs=model.input,outputs=model.layers[-1].output)
    
    #summarize the model
    print(model.summary())
    
    #extract features from each photo
    features = dict()
    for name in listdir(directory):
        #load image from file
        filename = directory + '/' + name
        image = load_img(filename,target_size=(224,224))
        #convert image pixels to a numpy array
        image = img_to_array(image)
        #reshape data for the model
        image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
        #prepare image for VGG mdodel
        image = preprocess_input(image)
        #get features
        feature = model.predict(image,verbose=0)
        #get image_id
        image_id = name.split('.')[0]
        #store feature
        features[image_id] = feature
        print('>%s' % name)
    
    return features

#extract features from all images
directory = 'C:/Users/admin/Downloads/imageCpation/Flickr8k_Dataset/Flicker8k_Dataset/'
features = extract_features(directory)
print('Extracted features : %d' %len(features))

#save to file
dump(features,open('C:/Users/admin/Downloads/imageCpation/features.pkl','wb'))

all_features = load(open('C:/Users/admin/Downloads/imageCpation/features.pkl','rb'))
for key in all_features.keys():
    print(all_features[key])
    print("--------------")
    