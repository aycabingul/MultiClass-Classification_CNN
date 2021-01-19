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

Kullanacağımız Cifar100 verisetini yükleyelim, değişkenlerimize atayalım ve .png uzantılı görüntüleri kaydetmek istediğimiz dosya yollarını belirtildi;

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

 şeklinde uyarı yazısı yazdırıldı;

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

Bu bölümün kodlarını incelemek için [tıklayınız.](https://github.com/aycabingul/MultiClass-Classification_CNN/blob/main/src/class_file.py)

## 2- Seçilen sınıfların görselleştirilmesi:

Bu bölümde eğitimde kullanacağımız sınıflara ait görüntüler görselleştirildi.

Bu bölümde kullanacağımız kütüphaneleri import edelim;

    import cv2 
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
Train ve test verilerimizin dosya yolları değişkene atandı ve sınıfların isimlerini içeren liste oluşturuldu:

    train_dir="../train"
    

6 sınıftan 10’ar örnek veriyi görselleştirmek için image_show fonksiyonu oluşturuldu. image_show fonksiyonu içinde ilk önce her sınıftan 10 veri okundu ve subplot ile yan yana çizdirildi daha sonra her sınıf için bir alt satıra inildi. Bu şekilde (6,10) bir subplot oluşturuldu. Son olarak her satırın başına sınıf ismi yazdırıldı:

    def class_show(train_path):
        concate_list=[]
        
        class_name=['/beaver','/boy','/forest','/oak_tree','/snail','/sunflowe']
    
        for x,name in enumerate(list_name):
            img_list=[]
            img_first=name+"/0.png"
            img_concate=cv2.imread(train_dir+img_first)
            a=x*10
            
            for i in range(10):
                img_name=name+"/"+str(i)+".png"
                img=cv2.imread(train_dir+img_name)
                ax2=plt.subplot(6,10,i+a
                if (i+1+a)==c:
                    ax2.set_ylabel(name[1:],rotation=0,labelpad=25)  
                    ax2.set_yticks([])
                    c=c+10
                else:
                    ax2.set_yticks([])
                    ax2.set_xticks([])
                          
                plt.imshow(img)
Yazdığımız `class_show` fonksiyonumuzu `train_path` parametresini girerek çalıştıralım:

    image_show(train_dir,list_name) 

Seçtiğimiz sınıflara ait örnek görüntüler aşağıdaki gibi olacak:
<p  align="center">
<img  src="https://i.hizliresim.com/S3Ao0q.png"  width="750">
</p>

Bu bölümün kodlarını incelemek için [tıklayınız.](https://github.com/aycabingul/MultiClass-Classification_CNN/blob/main/src/class_show.py)

---
## 3- Model Oluşturulması ve Eğitilmesi:

Bu bölümde **multi-class** classification problemimiz için bir **CNN modeli** oluşturuldu. **Loss** ve **accuracy** değerleri, oluşturulan grafikler ile incelendi.

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

    train_dir='../train'
    test_dir='../test'

İlk önce model tanımlandı:

    model=models.Sequential()

Modelimizin ilk katmanında;
- 32 hücreli,
- (3,3) boyutunda kernel’e sahip,
- kenarlara dolgulama işlemi uygulanması için **padding=’same’**, 
- aktivasyon fonksiyonu relu,
- lk katman olduğu için giriş görüntüsünün boyutu (32,32,3) olacak şekilde parametreler belirlenerek bir Conv2D layer’ı oluşturuldu:


    model.add(layers.Conv2D(32,(3,3),
                            padding='same',
                            activation='relu',
                            input_shape=(32,32,3)))

İkinci katman olarak, aynı parametreleri içeren bir Conv2D layer’ı oluşturuldu. Ancak önceki katmandan farklı olarak, bu bir ara katman olduğu için **input_shape** parametresi belirtilmedi:

    model.add(layers.Conv2D(32,(3, 3),
                             padding='same',
                             activation='relu'))
Görüntü boyutunu yarıya indirmek için (2, 2) boyutunda bir MaxPooling2D işlemi uygulandı:

    model.add(layers.MaxPooling2D((2, 2)))
Üçüncü katman olarak 64 hücreli, padding=’same’ seçilen, aktivasyon fonksiyonu relu olan Conv2D layer’ı oluşturuldu.

    model.add(layers.Conv2D(64,(3, 3),
                             padding='same',
                             activation='relu'))
Dördüncü katman olarak üçüncü katmandaki parametrelerin aynısı kullanılarak Conv2D layer’ı oluşturuldu:

    model.add(layers.Conv2D(64,(3,3),
                            padding='same',
                            activation='relu'))
Görüntü boyutunu yarıya indirmek için (2, 2) boyutunda bir MaxPooling2D işlemi uygulandı:

    model.add(layers.MaxPooling2D((2, 2)))
Beşinci katman olarak hücre sayısı 128 olan bir Conv2D layer’ı oluşturuldu:

    model.add(layers.Conv2D(128,(3,3),
                               padding='same',
                               activation='relu'))
Altıncı katman olarak beşinci katmandaki parametrelerin aynısı kullanılarak Conv2D layer’ı oluşturuldu:

    model.add(layers.Conv2D(128,(3,3),
                               padding='same',
                               activation='relu'))
Görüntü matrisimizi dense layer'a uygun hale getirmek, yani matrisi vektör haline dönüştürmek için **Flatten()** işlemi uygulandı:

    #Dense layer:
    model.add(layers.Flatten())
Modelimize, 256 hücreli, aktivasyon fonksiyonu olarak relu kullanan bir Dense layer eklendi:

    model.add(layers.Dense(256, activation='relu'))
Modelimize, 256 hücreli, aktivasyon fonksiyonu relu olan bir Dense layer daha eklendi:

    model.add(layers.Dense(256, activation='relu'))

Modelimizdeki çıkış katmanı olarak, tahmin edilecek 6 sınıfımız olduğu için, 6 hücreli ve **multi-class classification** problemi çözmeye çalıştığımız için aktivasyon fonksiyonu olarak **softmax** kullanan bir **Dense layer** eklendi:

    model.add(layers.Dense(6, activation='softmax'))
Modeli compile ederken, modelimizin çıkış katmanında softmax kullandığımız için loss fonksiyonu olarak **categorical_crossentropy** kullanıldı. Optimizer olarak **Adamax** ve **learning rate** parametresini **5e-3** olarak belirlendi. Accuracy değerini incelemek için metrics parametresinde acc kullanıldı:

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizers.Adamax(lr=5e-3), 
                  metrics=['acc'])
Dizayn edilen modeli incelemek için **model.summary()** fonksiyonu kullanıldı:

    print(model.summary())

<p  align="center">
<img  src="https://i.hizliresim.com/JnYKw6.png"  width="750">
</p>

Verilerimizi okurken pikselleri 0-1 aralığında normalize etmek için ImageDataGenerator fonksiyonunda rescale parametresine 1./255 değeri verildi:

    train_datagen=ImageDataGenerator(rescale=1./255)
    test_datagen= ImageDataGenerator(rescale=1./255)

Train ve test verilerimizi dosyalardan okumak için **flow_from_directory** fonksiyonunu kullanıldı. Bu fonksiyon, dosya yolundaki klasörlerden classları tespit eder ve labelları oluşturur. **Batch_size=20** parametresi ile de verileri 20’şer 20’şer okur. Görüntülerimizin giriş boyutu ise (32, 32) olarak belirlendi:

    train_generator = datagen_aug.flow_from_directory(train_dir,
                                              target_size = (32,32),
                                              batch_size =20,
                                              class_mode = 'categorical')
     
    test_generator = datagen_aug.flow_from_directory(test_dir,
                                              target_size = (32,32),
                                              batch_size = 20,
                                              class_mode = 'categorical')


Modelimizi okumak için  **flow_from_directory** kullanıldığı için eğitirken de **fit_generator** kullanıldı.
- Train_generator’den gelen veriler eğitim veri seti olarak kullanıldı.
- Her epoch’da 150 kez parametreleri güncellemesi için **step_per_epoch=150** olarak belirlendi.
- odelin verilerimizi 30 defa görmesi için **epoch=30** olarak belirlendi.
- Validation için **test_generator**'den gelen veriler kullanıldı:

Modeli eğitelim;

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=150,
                                  epochs=30,
                                  #callbacks=[csv_logger],
                                  validation_data=test_generator,
                                  validation_steps=30) 

<p  align="center">
<img  src="https://i.hizliresim.com/XFl7T3.png"  width="750">
</p>

Eğitilmiş modelimiz cifar100.h5 dosyasına kaydedildi:

    model.save('cifar100.h5')

Eğittiğimiz modelimizin sonuçlarını gözlemlemek için **Training and Validation Accuracy** ve **Training and Validation Loss** grafikleri çizdirildi: 

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
      plt.xlabel("Epoch")
      plt.ylabel("Acc")
      plt.title("Training and Validation Accuracy")
    
      plt.subplot(2,1,2)
      plt.plot(epochs, loss, "bo", label="Training loss")
      plt.plot(epochs, val_loss, "b", label="Validation loss")
      plt.title("Training and Validation Loss")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.legend()
      fig.tight_layout()
      plt.show()

      
    
    plot_acc_loss(history)


<p  align="center">
<img  src="https://i.hizliresim.com/wAd9Yj.png"  width="750">
</p>

Modelimiz validation verilerimizde %78.5 başarı elde etti.


<p  align="center">
<img  src="https://i.hizliresim.com/7VDYcc.png"  width="750">
</p>

Grafiğimizi incelediğimizde validation ve train loss arasındaki fark git gide arttığı için **overfitting** olduğu sonucuna varıldı.
Overfitting'i önlemek amacıyla sırasıyla **dropout** ve **data augmentation** yöntemleri uygulandı.

Bu bölümün kodlarını incelemek için [tıklayınız.](https://github.com/aycabingul/MultiClass-Classification_CNN/blob/main/src/model.py)

---

## 4- Dropout

**Dropout**, eğitim sırasında parametre olarak verilen oranda bağlantıları koparır. Overfitting'i önlemek için kullanılır.
Son Conv2D katmanından sonra, Dense Layer'dan önce ve iki Dense Layer arasında bağların yarısını koparmak için dropout eklendi.

    model.add(layers.Flatten())
    
    model.add(layers.Dropout(0.5)) <-------------- Dropout
    
    #Dense layer:
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))
    

Dropout eklendikten sonra model yeniden eğitildi ve eğitim sonucunda oluşan yeni grafikler ve değerler tekrar incelendi:
<p  align="center">
<img  src="https://i.hizliresim.com/5HvIqT.png"  width="750">
</p>

Modelimiz dropout işlemi sonrasında  validation verilerimizde %79.8 başarı elde etti.

<p  align="center">
<img  src="https://i.hizliresim.com/IRiZip.png"  width="750">
</p>

Grafikte incelediğimiz değerlere göre validation ve train loss’ları arasındaki fark biraz azaldığı için overfitting’e olumlu yönde bir etki ettiği görüldü. Ama daha iyi sonuç vermesi için **data augmentation** uygulandı.


---
## 5- Data Augmentation

Modelin iyi bir şekilde öğrenebilmesi için yeterli verimizin olması gerekmektedir. Yeteri kadar verimiz yoksa veri artırma yöntemlerine başvurmalıyız. **Data augmentation** yöntemi ile train verilerimizi çeşitlendirip modelimizin daha iyi öğrenmesi sağlandı. 

ImageDataGenerator fonksiyonu ile verilerimize **zoom_range**, **horizontal_flip** ve **rescale** işlemleri uygulandı:
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

Data augmentation işlemi sonrasında modelimiz yeniden eğitildi:

<p  align="center">
<img  src="https://i.hizliresim.com/qUxePw.png"  width="750">
</p>

Modelimiz, validation verilerimizde dropout ve data augmentation uygulaması sonrasında  %85.66 başarı elde etti.

<p  align="center">
<img  src="https://i.hizliresim.com/S3ol6N.png"  width="750">
</p>

Grafikte incelediğimiz değerlere göre modelimizde data augmentation ve dropout işlemleri uygulandıktan sonra train ve validation loss arasındaki farkın neredeyse kapandığını gördük. Overfitting için yapmış olduğumuz bu adımlar modeli iyileştirmemize yardımcı olmuştur.

---

## 6- est verilerinde predict işlemi:

Bu bölümde, eğittiğimiz model test verilerimizden her sınıftan bir örnek ile test edildi.

Kullanacağımız kütüphaneler import edildi:

    from keras.preprocessing import image
    from keras import models 
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2 
    import os

Sınıflarımızın isimlerini içeren list_name listesi, eğittiğimiz son model olan cifar100.h5 dosyası ve test verilerimiz değişkenlere atandı:

    list_name=['/beaver','/boy','/forest','/oak_tree','/snail','/sunflower']
    model=models.load_model('cifar100.h5')
    test_dir='/mnt/sdb2/ders/deep_learning_bsm/DeepLearning/Cifar100/test'

Her sınıf için test verileri arasından np.random.randint fonksiyonu ile rastgele birer görüntü seçip, seçtiğimiz görüntüyü eğitilmiş modele vererek tahmin edilen ve olması gereken sınıf görselleştirildi:

    for x,name in enumerate(list_name):
        predict_list=[]
        random=np.random.randint(1,100)
        path=(test_path+name+'/'+str(random)+".png")
        Giris1=image.load_img(path, target_size=(32,32))
        #numpy dizisine dönüştür
        Giris=image.img_to_array(Giris1)
        #görüntüyü ağa uygula
        y=model.predict(Data.reshape(1,32,32,3))
        #En yüksek tahmin sınıfını bul
        tahmin_indeks=np.argmax(y)
        
        ax3 = plt.subplot(6,6,x+1)
        ax3.set_yticks([])
        ax3.set_xticks([])
    
        ax3.set_xlabel('label:{0}\npred:{1}'.format(list_name[x][1:],list_name[tahmin_indeks[1:]))
        plt.imshow(Giris1)
    
    plt.show()


<p  align="center">
<img  src="https://i.hizliresim.com/wyo81x.png"  width="500">
</p>

Modelimiz 6 sınıf içerisinden 4 tanesini doğru tahmin etmiştir. Veri seti incelendiğinde bazı verilerin çok net olmadığı ve anlaşılmadığı gözlemlendi. Bu yüzden belli bir başarı yüzdesinin üzerine çıkmak için farklı sınıflar ile eğitildiğinde daha iyi sonuçlar verdiği tespit edildi.

Bu bölümün kodlarını incelemek için [tıklayınız.](https://github.com/aycabingul/MultiClass-Classification_CNN/blob/main/src/test.py)
