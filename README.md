*Türkçe*
# Metin Duygu Analizi

Proje kapsamında doğal dil işleme modelini kullanarak metinlerin duygu analizinin yapılması ve olumluluk durumlarının tespit edilmesi amaçlanmıştır. Natural Language Toolkit, Scikit-Learn, Numpy ve Pandas kütüphaneleri ile Flask modülü kullanılmıştır. 

NLTK kütüphanesinin içerisinde bulunan "stopwords" dil kaynaklı durak kelimeleri ile birlikte veri seti incelenmiş ve duygu analizi aşamasının hata payı büyük oranda düşürülmüştür.

Proje kapsamında bir Flask modülü kullanılmış olup web sunucusu üzerinde makine öğreniminin sonuçları bir karmaşıklık matrisi ile hesaplanıp grafiğe dönüştürülmüştür.

# Kullanılan Veri Seti

Proje içerisindeki **"Restaurant_Reviews.csv"** dosyası, proje içerisinde kullanılan veri setidir ve veriler rastgele restaurantlardan toplanan kullanıcı incelemelerini içermektedir. 

Toplanan veriler temizlenip düzenlenerek Pandas kütüphanesi ile en verimli çalışacak şekilde evrilmiştir.

**MIT lisanlı** proje dahilinde bu veri seti farklı projelerde uygun atıflar ile kullanılabilir.

🥴

*English*
# Text Sentiment Analysis

The aim of the project is to perform sentiment analysis on texts and determine the positivity status using a natural language processing model. The project utilizes the Natural Language Toolkit (NLTK), Scikit-Learn, Numpy, and Pandas libraries, along with the Flask module.

Within the NLTK library, the dataset was examined in conjunction with the "stopwords," which are language-specific stop words, and this significantly reduced the error rate in the sentiment analysis phase.

A Flask module is used in the project to calculate the results of machine learning on a web server and transform them into a complexity matrix and a graph.

# Used Dataset

The **"Restaurant_Reviews.csv"** file within the project is the dataset used in the project, which contains user reviews collected from random restaurants.

The collected data has been cleaned and organized to work efficiently using the Pandas library.

This dataset, included in the project under the **MIT license**, is available for use with proper attribution in different projects.
