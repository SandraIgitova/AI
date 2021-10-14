# Отчет по лабораторной работе 
## по курсу "Искусственый интеллект"

## Нейросетям для распознавания изображений


### Студенты: 

| ФИО       | Роль в проекте                     | Оценка       |
|-----------|------------------------------------|--------------|
| Васильева В.Е. | Подготовка и обработка данных |          |
| Савров Н.С. | Реализация сверточной нейросети |       |
| Павлова К.А.| Подготовка данных, написание отчета |      |

## Результат проверки

| Преподаватель     | Дата         |  Оценка       |
|-------------------|--------------|---------------|
| Сошников Д.В. |              |     3.5          |

> *Опоздание с сдаче*

## Тема работы

Необходимо подготовить набор данных и построить несколько нейросетевых классификаторов для распознавания рукописных символов (символы принадлежности множеству, пересечения, объединения множеств и пустого множества).

## Распределение работы в команде

## Подготовка данных

Исходники с рукописными символами:
![a1](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/affiliation_1.jpg)
![a2](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/affiliation_2.jpg)
![a3](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/affiliation_3.jpg)
![c1](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/crossing_1.jpg)
![c2](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/crossing_2.jpg)
![c3](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/crossing_3.jpg)
![e1](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/empty_1.jpg)
![e2](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/empty_2.jpg)
![e3](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/empty_3.jpg)
![u1](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/union_1.jpg)
![u2](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/union_2.jpg)
![u3](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-crabs/blob/master/img/union_3.jpg)

Все изображения были уменьшены до необходимого размера. Всем картинкам были присвоены номера класса. С шагом в 32х32 вырезались все объекты для выборки из больших картинок.

    def pars(objects, names, class_name, resized):
      for i in range(0, 320, 32):
        for j in range(0, 320, 32):
          cropped = resized[i: i + 32, j: j + 32]
          objects.append(cropped)
          names.append(class_name)

При подготовке датасета не читались рукописные символы с тонкими линиями, они были обведены повторно.

## Загрузка данных

    def compr(name):
      img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
      final_wide = 320
      r = float(final_wide) / img.shape[1]
      dim = (final_wide, int(img.shape[0] * r))
      
      resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
      cv2_imshow(resized)
      print(resized.shape)
      return resized

Иcпользовалась библиотека opencv.

## Обучение нейросети

### Полносвязная однослойная сеть

    import tensorflow as tf

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(4,activation='softmax',input_shape=(32*32*3,)))

    model.compile(tf.keras.optimizers.Adam(0.1),'sparse_categorical_crossentropy',['accuracy'])
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_test, y_test) ,batch_size=4,epochs=10)

Model: "sequential_21"

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_36 (Dense)             (None, 4)                 12292     
    =================================================================
    Total params: 12,292
    Trainable params: 12,292
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/10
    255/255 [==============================] - 1s 2ms/step - loss: 55.8863 - accuracy: 0.5147 - val_loss: 97.9317 - val_accuracy: 0.3889
    Epoch 2/10
    255/255 [==============================] - 0s 2ms/step - loss: 31.3722 - accuracy: 0.6833 - val_loss: 58.5520 - val_accuracy: 0.5611
    Epoch 3/10
    255/255 [==============================] - 0s 2ms/step - loss: 44.7296 - accuracy: 0.7020 - val_loss: 29.5061 - val_accuracy: 0.7111
    Epoch 4/10
    255/255 [==============================] - 0s 2ms/step - loss: 31.5878 - accuracy: 0.7657 - val_loss: 30.4306 - val_accuracy: 0.7500
    Epoch 5/10
    255/255 [==============================] - 0s 2ms/step - loss: 55.9881 - accuracy: 0.7039 - val_loss: 22.7869 - val_accuracy: 0.8278
    Epoch 6/10
    255/255 [==============================] - 0s 2ms/step - loss: 30.5307 - accuracy: 0.8108 - val_loss: 57.5870 - val_accuracy: 0.8056
    Epoch 7/10
    255/255 [==============================] - 0s 2ms/step - loss: 29.7547 - accuracy: 0.8206 - val_loss: 58.6173 - val_accuracy: 0.6444
    Epoch 8/10
    255/255 [==============================] - 0s 2ms/step - loss: 28.5107 - accuracy: 0.8167 - val_loss: 32.8547 - val_accuracy: 0.7722
    Epoch 9/10
    255/255 [==============================] - 0s 2ms/step - loss: 24.8384 - accuracy: 0.8284 - val_loss: 34.6629 - val_accuracy: 0.8333
    Epoch 10/10
    255/255 [==============================] - 0s 2ms/step - loss: 34.0669 - accuracy: 0.8078 - val_loss: 36.5264 - val_accuracy: 0.7500

    <tensorflow.python.keras.callbacks.History at 0x7f337c1eea10>

Средняя точность на тестовой выборке около 80%, что не впечатляет.

### Свёрточная сеть

Сверточная нейронная сеть будет состоять из двух сверточных слоев размера 3х3 и одного maxpooling слоя размером 2х2. Также добавим 2 линейных слоя с активационными функциями relu и softmax. В качестве оптимизатора используем adam, в качестве функции ошибки sparse_categorical_crossentropy.

    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
    from keras.constraints import maxnorm

    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    epochs = 3
    lrate = 0.0005
    decay = lrate/epochs
    adam = tf.keras.optimizers.Adam(lr=lrate, decay=decay)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())

Model: "sequential_20"

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_28 (Conv2D)           (None, 32, 32, 32)        896       
    _________________________________________________________________
    dropout_28 (Dropout)         (None, 32, 32, 32)        0         
    _________________________________________________________________
    conv2d_29 (Conv2D)           (None, 32, 32, 32)        9248      
    _________________________________________________________________
    max_pooling2d_14 (MaxPooling (None, 16, 16, 32)        0         
    _________________________________________________________________
    flatten_14 (Flatten)         (None, 8192)              0         
    _________________________________________________________________
    dense_34 (Dense)             (None, 512)               4194816   
    _________________________________________________________________
    dropout_29 (Dropout)         (None, 512)               0         
    _________________________________________________________________
    dense_35 (Dense)             (None, 4)                 2052      
    =================================================================
    Total params: 4,207,012
    Trainable params: 4,207,012
    Non-trainable params: 0
    _________________________________________________________________
    None
    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py:375: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      "The `lr` argument is deprecated, use `learning_rate` instead.")

    # dataset = tf.data.Dataset.from_tensor_slices((ims, names))

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=4)

    Epoch 1/3
    255/255 [==============================] - 23s 87ms/step - loss: 0.7637 - accuracy: 0.7137 - val_loss: 0.0904 - val_accuracy: 0.9667
    Epoch 2/3
    255/255 [==============================] - 22s 87ms/step - loss: 0.0582 - accuracy: 0.9853 - val_loss: 0.0443 - val_accuracy: 0.9833
    Epoch 3/3
    255/255 [==============================] - 22s 87ms/step - loss: 0.0112 - accuracy: 0.9990 - val_loss: 0.0303 - val_accuracy: 0.9889

    <tensorflow.python.keras.callbacks.History at 0x7f337c3e46d0>

После первой же эпохи точность на валидационной выборке составила 96% и дальше выросла до 98%, что очень впечатляет.

Изначально использовалась функция потерь categorical_crossentropy, но возникали ошибки. Исправлено на sparse_categorical_crossentropy.
Также были проблемы с размерностью, так как простейшая  сеть принимала только размерность (32\*32\*3) а сверточная (32, 32,3).

## Выводы
Данная лабораторная работа была крайне интересна, несмотря на то, что при ее выполнении мы столкнулись с некоторыми трудностями вроде реализации простейшей сети на keras и подбором параметров для сверточной сети. Очень впечатляет, когда сеть работает не с какими то подготовленными данными, а с твоими собственными. Также было полезно применить навыки использования библиотеки opencv для работы с изображениями. Благодаря слаженным действиям участников команды, дистанционным собраниям и заинтересованности в задании, работа была выполнена довольно быстро.
