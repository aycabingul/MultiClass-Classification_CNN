#  Multi Class Classification with Convolutional Neural Network
-   100 sınıf içeren [Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html) veriseti içerisinden 6 sınıf seçildi ve Convolutional Neural Network kullanılarak sınıflandırma problemi çözüldü.
-   Model başarısını iyileştirmek için `Dropout`, `Data Augmentation` gibi yöntemler kullanıldı.

## 1- Veri yükleme ve sınıf seçimi:
İlk olarak cifar100 veri seti’ni yükledikten sonra **beaver**, **boy**, **forest**, **oak_tree**, **snail**, **sunflower** sınıfları cifar100 veri setinden çekildi.Train ve test klasörleri oluşturulduktan sonra içerisine bu sınıflara ait altı adet alt klasör oluşturularak, veriler bu klasörlere kaydedildi. 

Bu bölümde kullanacağımız kütüphaneleri import edildi;

    import json
    from matplotlib import pyplot
    import cv2
    import numpy as np
    import os

Kullanacağımız Cifar100 verisetini yükleyelim, değişkenlerimize atayalım ve .png uzantılı görüntüleri kaydetmek istediğimiz dosya yollarını belirtelim;

    # load data
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    train_dir='../train'#train verisi için dosya yolu
    test_dir='../test'#test verisi için dosya yolu

Klasörleri oluşturacak open_dir fonksiyonu oluşturuldu. Train ve test klasörlerini oluşturulması için open_dir fonksiyonu çağrıldı:

    def open_dir(data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    open_path(train_dir)
    open_path(test_dir)

Seçtiğimiz 6 class'ı klasörlere kaydetmek için kullanacağımız fonksiyonu oluşturalım. Öncelikle class isimlerini içeren bir `list_name` listesi ve class numaralarını içeren bir `list_num` listesi oluşturalım. Bu listelerdeki değerleri kullanarak class numaralarına göre ayrım yaparak klasörlere kaydedelim. Eğer bu klasörler önceden oluşturuldu ise, 

> "... konumunda dosya vardır kontrol ediniz"

 şeklinde uyarı yazısı yazdıralım;

    class_name=['/beaver','/boy','/forest','/oak_tree','/snail','/sunflowe']
    class_num=[4, 11, 33, 52, 77, 82]
    def class_selection(X_data,y_data,image_dir):
    	for i,num in enumerate(list_num):
            if not os.path.exists(image_dir+list_name[i]):
                index=np.where(y_data==num) #class_num'daki dataların indexlerini belirlemek
                subset_x_data=X_data[np.isin(y_data,[num]).flatten()]
                for a,x in enumerate(subset_x_data):
                    image_path=(image_dir+list_name[i])
                    open_dir(image_path)
                    image_path=(image_path+"/"+str(a)+".png")
                    cv2.imwrite(image_path,x)
        else:
            print(image_dir+list_name[i]+" konumunda dosya vardır kontrol ediniz")
    
    class_selection(X_train,y_train,train_dir)
    class_selection(X_test,y_test,test_dir)


Oluşturulan klasörler aşağıdaki gibi olacaktır;

<p  align="center">
<img  src="https://i.hizliresim.com/VpKUOF.png"  width="750">
</p>

Bu bölümün kodlarını incelemek için [tıklayınız.](https://github.com/recepayddogdu/MultiClass-Classification_CNN/blob/main/src/class_selection.py)

---https://hizliresim.com/VpKUOF
## 2- Seçilen sınıflara ait örnek verilerin görselleştirilmesi

Projenin bu bölümünde, eğitimde kullanacağımız **bee**, **couch**, **girl**, **lawn_mower**, **whale**, **wolf** sınıflarına ait örnek görüntüleri inceleyeceğiz.

Bu bölümde kullanacağımız kütüphaneleri import edelim;

    import cv2 
    import numpy as np
    import os
    import matplotlib.pyplot as plt
`train` ve `test` verilerimizin dosya yollarını belirtelim;

    train_path="data/train"
    test_path="data/test"

6 sınıftan 10'ar örnek veriyi grafik olarak çizmek için `class_show` adında bir fonksiyon oluşturalım. Sınıf isimlerimizi ve numaralarını `list_name` ve `list_num` değişkenlerine atayalım. Sonrasında her sınıf klasöründen alınan 10 adet görüntüyü `np.concatenate` fonksiyonu ile yan yana birleştirelim. Birleştirdiğimiz 6 sınıfa ait görüntüleri alt alta ekranda görüntüleyelim. Satır başına ise o satırdaki görüntülerin ait olduğu sınıf ismini yazdıralım;

    def class_show(train_path):
        concate_list=[]
        
        list_name=['/bee','/couch','/girl','/lawn_mower','/whale','/wolf']
        list_num=[6, 25, 35, 41, 95, 97]
    
        for x,name in enumerate(list_name):
            img_list=[]
            img_first=name+"/0.png"
            img_concate=cv2.imread(train_path+img_first)
            for i in range(10):
                img_name=name+"/"+str(i)+".png"
                img=cv2.imread(train_path+img_name)
                img_list.append(img)
                if i>0:
                    img_concate=np.concatenate((img_concate,img_list[i]),axis=1)
            concate_list.append(img_concate)
            ax3 =plt.subplot(10,1,x+1)
            ax3.set_yticks([])
            ax3.set_xticks([])
            ax3.set_ylabel(name[1:], rotation=0, labelpad=32)
            plt.imshow(concate_list[x])
            plt.axis('on')
        
        plt.show()
Yazdığımız `class_show` fonksiyonumuzu `train_path` parametresini girerek çalıştıralım:

    class_show(train_path)

Seçtiğimiz sınıflara ait örnek görüntüler aşağıdaki gibi olacak:
<p  align="center">
<img  src="https://i.hizliresim.com/uRxXG3.jpg"  width="750">
</p>

Bu bölümün kodlarını incelemek için [tıklayınız.](https://github.com/recepayddogdu/MultiClass-Classification_CNN/blob/main/src/class_show.py)

---
## 3- Model oluşturulması ve eğitimi

Projenin bu bölümünde multi-class clasification problemimiz için bir CNN modeli oluşturacağız ve modeli Google Colab'da eğiteceğiz. Loss ve accuracy değerlerini inceleyeceğiz.

Bu bölümde kullanacağımız kütüphaneleri import edelim:

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing import image
    from matplotlib import pyplot as plt
    from keras.datasets import cifar100
    from keras import layers
    from keras import models
    import tensorflow as tf
    import numpy as np
    import cv2
    import os
    from keras import optimizers
    from keras.callbacks import CSVLogger

`train` ve `test` verilerimizin dosya yollarını belirleyelim:

    train_dir='data/train'
    test_dir='data/test'

Modeli tanımlamakla başlayalım:

    model=models.Sequential()

Modelimizin 1. katmanında;
- 32 hücreli,
- (3, 3) boyutunda kernele sahip,
- kenarlarda padding dolgusu olan,
- aktivasyon fonksiyonu olarak relu kullanan,
- ilk katman olduğu için de giriş boyutu (32, 32, 3)

olacak şekilde parametreler belirterek, bir Conv2D layer'ı ekleyelim:

    model.add(layers.Conv2D(32,(3,3),
                            padding='same',
                            activation='relu',
                            input_shape=(32,32,3)))

Modelimizin 2. katmanında 1. katman ile aynı parametreler kullanıldı, önceki katmandan gelen verileri kullanacağı için `input_shape` belirtilmedi:

    model.add(layers.Conv2D(32,(3, 3),
                             padding='same',
                             activation='relu'))
Görüntü boyutunu yarıya indirmek için (2, 2) boyutunda bir `MaxPooling2D` işlemi uygulayalım:

    model.add(layers.MaxPooling2D((2, 2)))
Modelimizin 3. katmanında 64 hücreli bir Conv2D layer'ı ekleyelim:

    model.add(layers.Conv2D(64,(3, 3),
                             padding='same',
                             activation='relu'))
Modelimizin 4. katmanında 64 hücreli bir Conv2D layer'ı ekleyelim:

    model.add(layers.Conv2D(64,(3,3),
                            padding='same',
                            activation='relu'))
Görüntü boyutunu yarıya indirmek için (2, 2) boyutunda bir `MaxPooling2D` işlemi uygulayalım:

    model.add(layers.MaxPooling2D((2, 2)))
Görüntü matrisimizi dense layer'a uygun hale getirmek, matrisi vektör haline dönüştürmek için `Flatten()` işlemi uygulayalım:

    #Dense layer:
    model.add(layers.Flatten())
Modelimize, 256 hücreli, aktivasyon fonksiyonu olarak `relu` kullanan bir dense layer ekleyelim:

    model.add(layers.Dense(256, activation='relu'))

Modelimizdeki çıkış katmanı olarak, tahmin edilecek 6 sınıfımız olduğu için, 6 hücreli ve multi-class classification problemi çözmeye çalıştığımız için, aktivasyon fonksiyonu olarak `softmax` kullanan bir dense layer ekleyelim:

    model.add(layers.Dense(6, activation='softmax'))
Modeli compile ederken, modelimizin çıkış katmanında `softmax` kullandığımız için loss fonksiyonu olarak `categorical_crossentropy` kullanıyoruz. Optimizer olarak `Adamax` ve learning rate parametresini `5e-4` olarak belirliyoruz. Accuracy değerini incelemek için metrics parametresinde `acc` kullanalım:

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizers.Adamax(lr=5e-4), 
                  metrics=['acc'])
Oluşturduğumuz modeldeki katmanları ve parametreleri gözlemleyelim:

    print(model.summary())

<p  align="center">
<img  src="https://i.hizliresim.com/1sXeqV.jpg"  width="750">
</p>

Verilerimizi okurken değerleri 0-1 aralığında normalize etmek için `ImageDataGenerator` fonksiyonunda `rescale` parametresini kullanalım:

    train_datagen=ImageDataGenerator(rescale=1./255)
    test_datagen= ImageDataGenerator(rescale=1./255)

Train ve Test verilerimizi dosyalardan okumak için `flow_from_directory` fonksiyonunu kullanalım. Bu fonksiyon, dosya yolundaki klasörlerden classları tespit eder ve labelları oluşturur. Verilerin tümünü birden yüklememek için `batch_size=20`, görüntülerimizin giriş boyutunu ise `(32, 32)` olarak belirleyelim:

    train_generator = datagen_aug.flow_from_directory(train_dir,
                                              target_size = (32,32),
                                              batch_size =20,
                                              class_mode = 'categorical')
     
    test_generator = datagen_aug.flow_from_directory(test_dir,
                                              target_size = (32,32),
                                              batch_size = 20,
                                              class_mode = 'categorical')


Modelimizi eğitmek için `fit_generator` kullanıyoruz.
- Eğitim verileri olarak `train_generator`'den gelen verileri kullanıyoruz.
- Her epoch'da 150 kez parametreleri güncellemek için 		 `steps_per_epoch=150` olarak belirliyoruz.
- Verilerimizin 30 defa modeli dolaşması için `epoch=30` olarak belirliyoruz.
- Validation için `test_generator`'den gelen verileri kullanıyoruz.

Modeli eğitelim;

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=150,
                                  epochs=30,
                                  #callbacks=[csv_logger],
                                  validation_data=test_generator,
                                  validation_steps=30) 

<p  align="center">
<img  src="https://i.hizliresim.com/F23SGm.jpg"  width="750">
</p>

Eğitilen modeli .h5 uzantılı dosyaya kaydedelim:

    model.save('cifar100-son/model/best_model_end.h5')

Eğittiğimiz modelimizin sonuçlarını gözlemlemek için *Training and Validation Accuracy* ve *Training and Validation Loss* grafiklerini çizdirelim:

    def plot_acc_loss(x):  
      acc = x.history["acc"]
      val_acc = x.history["val_acc"]
      loss = x.history["loss"]
      val_loss = x.history["val_loss"]
      print("acc =", acc[-1])
      print("val_acc = ", val_acc[-1])
      print("loss =", loss[-1])
      print("val_loss =", val_loss[-1])
      epochs = range(1, len(acc) + 1)
      fig = plt.figure()
      plt.subplot(2,1,1)
      plt.plot(epochs, acc, "bo", label="Training acc")
      plt.plot(epochs, val_acc, "b", label="Validation acc")
      plt.xlabel("Epochs")
      plt.ylabel("Accuracy")
      plt.title("Training and Validation Accuracy")
    
      plt.subplot(2,1,2)
      plt.plot(epochs, loss, "bo", label="Training loss")
      plt.plot(epochs, val_loss, "b", label="Validation loss")
      plt.title("Training and Validation Loss")
      plt.xlabel("Epochs")
      plt.ylabel("Loss")
      plt.legend()
      fig.tight_layout()
      plt.show()
      fig.savefig("cifar100-son/graph.png")
      
    
    plot_acc_loss(history)


<p  align="center">
<img  src="https://i.hizliresim.com/jb3XPT.jpg"  width="750">
</p>

Modelimiz test verilerimizde **`%82.8`** başarı elde etti.


<p  align="center">
<img  src="https://i.hizliresim.com/0UnOSG.png"  width="750">
</p>

Grafikte incelediğimiz değerlere göre modelimizin **overfitting** olduğu sonucuna varıyoruz.

Overfitting'i önlemek amacıyla sırasıyla `dropout` ve `data augmentation` yöntemlerini uygulayacağız.

Bu bölümün kodlarını incelemek için [tıklayınız.](https://github.com/recepayddogdu/MultiClass-Classification_CNN/blob/main/src/best_model.py)

---

## 4- Dropout

**`Dropout`**, eğitim sırasında parametre olarak verilen oranda, bağlantıları koparır. Overfitting'i önlemek için kullanılır.

Son Conv2D katmanından sonra, Dense Layer'dan önce bağların yarısını koparmak için `dropout` ekleyelim:

    model.add(layers.Conv2D(64,(3,3),
                            padding='same',
                            activation='relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Dropout(0.5)) <-------------- Dropout
    
    #Dense layer:
    model.add(layers.Flatten())
    
    model.add(layers.Dense(256, activation='relu'))

`Dropout` eklendikten sonra modeli yeniden eğitelim, eğitim sonucunda oluşan yeni grafikleri ve değerleri tekrar inceleyelim:
<p  align="center">
<img  src="https://i.hizliresim.com/KRveeO.jpg"  width="750">
</p>

Modelimiz `dropout` işlemi sonrasında test verilerimizde **`%83.8`** başarı elde etti. `Dropout` uygulamadan önce başarı oranı **`%82.8`** idi.

<p  align="center">
<img  src="https://i.hizliresim.com/leHvCc.png"  width="750">
</p>

Grafikte incelediğimiz değerlere göre modelimizin **overfitting** durumunda az da olsa iyileşme görüyoruz. Ancak hala modelimiz çok iyi durumda değil.

Bu durumu daha da iyileştirmek için **data augmentation** uygulayacağız.

Bu bölümün kodlarını incelemek için [tıklayınız.](https://github.com/recepayddogdu/MultiClass-Classification_CNN/blob/main/src/best_model.py)

---
## 5- Data Augmentation

**Data augmentation** yöntemi ile **train** verilerimizi çeşitlendirip modelimizi iyileştirmeyi hedefliyoruz.

`ImageDataGenerator` fonksiyonu ile verilerimize `zoom_range`, `horizontal_flip` ve `rescale` işlemlerini uygulayacak generator'u yazıyoruz:

    datagen = ImageDataGenerator(
                  zoom_range=0.2,
                  horizontal_flip=True,
                  rescale=1./255)
`Train` ve `test` verilerimize data augmentation'ı uygulayalım:

    train_generator = datagen.flow_from_directory(train_dir,
                                              target_size = (32,32),
                                              batch_size =20,
                                              class_mode = 'categorical')
     
    test_generator = datagen.flow_from_directory(test_dir,
                                              target_size = (32,32),
                                              batch_size = 20,
                                              class_mode = 'categorical')

Data augmentation işlemi sonrasında modelimizi yeniden eğitiyoruz.

<p  align="center">
<img  src="https://i.hizliresim.com/jM8yPN.jpg"  width="750">
</p>

Modelimiz augmentation ve dropout işlemleri sonrasında **test** verilerimizde **`%84.8`** başarı elde etti. Bu işlemleri uygulamadan önce başarı oranı **`%82.8`** idi.

<p  align="center">
<img  src="https://i.hizliresim.com/3Pp6Ts.png"  width="750">
</p>

Grafikte incelediğimiz değerlere göre modelimizde **data augmentation** ve **dropout** işlemleri uygulamadan önceki haline göre iyileşme olduğunu görüyoruz. Validation ve Train değerleri birbirine yaklaştığı için **overfitting oluşmadığı sonucuna varıyoruz.**

Bu bölümün kodlarını incelemek için [tıklayınız.](https://github.com/recepayddogdu/MultiClass-Classification_CNN/blob/main/src/best_model.py)

---

## 6- Eğittiğimiz model ile test verilerinde predict işlemi

Projenin bu bölümünde, eğittiğimiz modeli test verilerimizdeki her sınıftan bir örnek ile test edeceğiz.

Bu bölümde kullanacağımız kütüphaneleri import edelim:

    from keras.preprocessing import image
    from keras import models 
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2 
    import os

Sınıflarımızın isimlerini içeren `list_name` listesini, eğittiğimiz son model olan `best_model_end.h5` dosyasını ve test verilerimizi değişkenlere atayalım:

    list_name=['/bee','/couch','/girl','/lawn_mower','/whale','/wolf']
    model=models.load_model('src/models/best_model_end.h5')
    test_path='data/test'

Her sınıftaki 100 adet test verisi arasından rastgele birer data seçip, seçtiğimiz datanın olması gereken sınıfı ile modelimizin tahmin ettiği sınıfı karşılaştıralım:

    for x,name in enumerate(list_name):
        predict_list=[]
        random=np.random.randint(1,100)
        path=(test_path+name+'/'+str(random)+".png")
        Data1=image.load_img(path, target_size=(32,32))
    
        Data=image.img_to_array(Data1)
        
        y=model.predict(Data.reshape(1,32,32,3))
        
        predict_ind=np.argmax(y)
        
        ax3 = plt.subplot(6,1,x+1)
        ax3.set_yticks([])
        ax3.set_xticks([])
    
        ax3.set_ylabel('Olmasi Gereken:{0}\nTahmin:{1}'.format(list_name[x][1:],
                                                        list_name[predict_ind][1:]),
                                                        rotation=0,
                                                        labelpad=75)
        plt.imshow(Data1)
    
    plt.show()


<p  align="center">
<img  src="https://i.hizliresim.com/gUNjQs.png"  width="500">
</p>

Yukarıdaki görselde göreceğimiz üzere bee, girl, lawn_mower, wolf sınıflarımız doğru tahmin edildi, couch ve whale sınıflarımız doğru tahmin edilemedi. **Modelimiz 6 sınıftan 4 tanesini doğru tahmin etti.** Test sonuçlarımızda da **`%84.5`** doğruluk sonucuna ulaşmıştık.

Bu bölümün kodlarını incelemek için [tıklayınız.](https://github.com/recepayddogdu/MultiClass-Classification_CNN/blob/main/src/test.py)
