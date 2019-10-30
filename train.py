import loadData
import model
from keras.callbacks import ModelCheckpoint

#load training images
filename = 'C:/Users/admin/Downloads/imageCpation/Flickr8k_text/Flickr_8k.trainImages.txt'
train = loadData.load_set(filename)    
print("Train Dataset: %d " %len(train))

#load training descriptions
filename2 = 'C:/Users/admin/Downloads/imageCpation/Flickr8k_text/descriptions.txt'
train_descriptions = loadData.load_clean_descriptions(filename2,train)
print("Training Descriptions : %d " %len(train_descriptions))

#load photo features
filename3 = 'C:/Users/admin/Downloads/imageCpation/features.pkl'
train_features = loadData.load_photo_features(filename3,train)    
print("Photos: train = %d" %len(train_features))    

#prepare tokenizer
tokenizer = loadData.create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary size : %d" %vocab_size)  

#determine maximum sequence length
max_length = loadData.max_length(train_descriptions)
print('Description Length: %d' % max_length)

#prepare sequences
X1train,X2train,ytrain = loadData.create_sequences(tokenizer,max_length,train_descriptions,train_features,vocab_size)

#dev Dataset (Kind of validation)
filename4 = 'C:/Users/admin/Downloads/imageCpation/Flickr8k_text/Flickr_8k.devImages.txt'
test = loadData.load_set(filename4)
print("Test Dataset : %d " %len(test))

#load descriptions
filename5 = 'C:/Users/admin/Downloads/imageCpation/Flickr8k_text/descriptions.txt'
test_descriptions = loadData.load_clean_descriptions(filename2,test)
print("Test Descriptions : %d " %len(test_descriptions))

#load photo features
filename3 = 'C:/Users/admin/Downloads/imageCpation/features.pkl'
test_features = loadData.load_photo_features(filename3,test)    
print("Photos: train = %d" %len(test_features))    

#prepare tokenizer
#tokenizer = loadData.create_tokenizer(test_descriptions)
#vocab_size = len(tokenizer.word_index) + 1
#print("Vocabulary size : %d" %vocab_size)  

#prepare the sequences
X1test,X2test,ytest = loadData.create_sequences(tokenizer,max_length,test_descriptions,test_features,vocab_size)

#define the model
model = model.define_model(vocab_size,max_length)

#define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')

#fit the model
model.fit([X1train,X2train],ytrain,epochs=20,verbose=2,callbacks=[checkpoint],validation_data=([X1test,X2test],ytest))

