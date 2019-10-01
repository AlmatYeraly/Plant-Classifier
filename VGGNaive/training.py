### Almat Yeraly and Jude Battista
### This was directly copied and modified from https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.summary()


model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

history = model.fit_generator(train_generator,
                                steps_per_epoch=ntrain // batch_size,
                                epochs = 50,
                                validation_data = val_generator,
                                validation_steps = nval // batch_size)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'b', label='training accuracy')
plt.plot(epochs, val_acc, 'r', label='validation accuracy')
plt.title('training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='training loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.title('training and validation loss')
plt.legend()

plt.show()


def read_and_process_image(list_of_images):
    X = []
    y = []

    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), 
                (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
        if 'aloe' in image:
            y.append(1)
        elif 'peace' in image:
            y.append(0)

    return X, y

X_test, y_test = read_and_process_image(test_imgs)
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale = 1./255)

pred = model.predict(test_datagen.flow(x, batch_size=1)[0])

resultCategories = []
resultEnum = ['aloe vera', 'peace lilly', 'spider plant', 'dumb cane']
columns = 5
i = 0
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    print(pred)
    maxPred = 0
    maxNdx = -1
    for ndx, prediction in enumerate(pred[0]):
        print(prediction)
        if prediction > maxPred:
            maxPred = prediction
            maxNdx = ndx
    resultCategories.append(resultEnum[maxNdx])
    plt.subplot(20 / columns + 1, columns, i+1)
    plt.title('this is ' + resultCategories[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i%20 ==0:
        break
plt.show()

'''
columns = 5
i = 0
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('aloevera')
    else:
        text_labels.append('peacelily')
    plt.subplot(20 / columns + 1, columns, i+1)
    plt.title('this is ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i%10 ==0:
        break
plt.show()
'''
