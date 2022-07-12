import pandas as pd
import torch
import pickle
from transformers import AutoTokenizer, AutoModel

def inferens(scv):
    """"
        Функция для инференса модели.
        На вход подается .csv файл, или ссылка на его расположение в дирректории.
        Сам .csv файл имеет вид: [«text»: «входной текст», «class_true»: «истинный лейбл»]

        return: pd.DataFrame [«text», «class_true», «class_prediction», «probabilities»]
    """

    df = pd.read_csv(scv, sep=',', engine='python')
    # извлечение коментариев из датафрейма и формирование списка коментариев
    sentences_test = list(df.text.values)

    # инициализация токенайзера и модели предобученными весами
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
    model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

    # токенизация батче размером 64 и дальнейшее получения их эмбедингов, с их последующей нормализацией
    encoded_input= tokenizer(sentences_test, truncation=True, padding=True, max_length=64, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.pooler_output
    df_embeddings = torch.nn.functional.normalize(embeddings)
    # инициализация предобученной логистической регрессии
    lr_clf = pickle.load(open('LogisticRegression_model.sav', 'rb'))

    probs = lr_clf.predict_proba(df_embeddings)[:, 1]
    predict_labels = lr_clf.predict(df_embeddings)
    df['class_prediction'] = predict_labels
    df['probabilities'] = probs

    return df