import os
import pandas as pd
import glob

# Hikayeleri ve yazarları tutacak listeler
texts = []
authors = []

# Ana klasör yolunu belirt
dataset_path =  "proje/dataset_authorship" 


# Her yazar klasörüne bak
for author_folder in os.listdir(dataset_path):
    author_path = os.path.join(dataset_path, author_folder)
    if os.path.isdir(author_path):  # Eğer gerçekten klasörse
        # O yazarın klasöründeki her .txt dosyasını oku
        for file_path in glob.glob(os.path.join(author_path, "*.txt")):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                texts.append(content)
                authors.append(author_folder)

# Bir DataFrame oluştur
df = pd.DataFrame({'author': authors, 'text': texts})

# İlk 5 satıra bakalım
print(df.head())









import re

def temizle(metin):
    # Küçük harfe çevir, özel karakterleri kaldır
    metin = metin.lower()
    metin = re.sub(r"[^a-zçğıöşü\s]", "", metin)  # Sadece harfler ve boşluklar
    return metin

# Temizlenmiş yeni bir sütun ekle
df['clean_text'] = df['text'].apply(temizle)

print(df[['author', 'clean_text']].head())







from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF vektörü oluştur
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

# Etiketleri al
y = df['author']












from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)






from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def modeli_degerlendir(model_adi, y_test, y_pred):
    print(f"\n{model_adi} Sonuçları:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))








from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

modeli_degerlendir("SVM", y_test, y_pred_svm)









from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

modeli_degerlendir("Random Forest", y_test, y_pred_rf)










from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

modeli_degerlendir("Decision Tree", y_test, y_pred_dt)









from xgboost import XGBClassifier

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

modeli_degerlendir("XGBoost", y_test, y_pred_xgb)







from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

modeli_degerlendir("Naive Bayes", y_test, y_pred_nb)









from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(max_iter=300)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)

modeli_degerlendir("MLP", y_test, y_pred_mlp)







from sklearn.feature_extraction.text import TfidfVectorizer

# 2-gram
vectorizer_bigram = TfidfVectorizer(ngram_range=(2, 2))
X_bigram = vectorizer_bigram.fit_transform(df['clean_text'])

# 3-gram
vectorizer_trigram = TfidfVectorizer(ngram_range=(3, 3))
X_trigram = vectorizer_trigram.fit_transform(df['clean_text'])













from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def egit_degerlendir(X, y, model_adi="Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{model_adi} Sonuçları:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))










# 2-gram ile model eğitimi
egit_degerlendir(X_bigram, y, "Kelime 2-gram")

# 3-gram ile model eğitimi
egit_degerlendir(X_trigram, y, "Kelime 3-gram")












# Karakter 2-gram
vectorizer_char2 = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
X_char2 = vectorizer_char2.fit_transform(df['clean_text'])

# Karakter 3-gram
vectorizer_char3 = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
X_char3 = vectorizer_char3.fit_transform(df['clean_text'])








egit_degerlendir(X_char2, y, "Karakter 2-gram")
egit_degerlendir(X_char3, y, "Karakter 3-gram")

