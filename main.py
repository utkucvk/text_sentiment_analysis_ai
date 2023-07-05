import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv')

import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

#VERİ ÖNİŞLEME
derlem = []
for i in range(986):
    yorum = re.sub('[^a-zA-Z]',' ', yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) 
             for kelime 
             in yorum 
             if not kelime 
             in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)

# for döngüsü ile birlikte bütün yorumlar için bir filtre fonksiyonu oluşturuldu. her bir yorum satırı için teker teker derleme yapıldı.

#ÖZNİTELİK ÇIKARIMI
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000) #rami rahatlatmak amacı ile yalnızca en çok kullanılan 1000 kelimeyi kullan
X = cv.fit_transform(derlem).toarray() #bağımsız değişken
y = yorumlar.iloc[:,1].values #bağımlı değişken

#MAKİNE ÖĞRENMESİ
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm) # %68 başarı oranı / %32 hata payı


from flask import Flask, render_template
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def plot_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
    plt.colorbar()
    
    classes = ['Negatif', 'Pozitif']
    tick_marks = np.arange(len(classes))
    
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.0
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel("Gerçeklik Matrisi")
    plt.xlabel("Tahmini Değer")
    
@app.route('/')
def home():
        
    cm = np.array([[48, 42], [20,88]])
    plot_confusion_matrix(cm)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    
    return render_template('index.html', image=string.decode('utf-8'))
if __name__ == '__main__':
    app.run()
