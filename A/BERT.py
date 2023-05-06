import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification, TFBertModel
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

train_df = pd.read_csv('train.csv')

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(train_df, test_size=0.2, random_state=141)

model_name = 'bert-base-multilingual-cased'

tokenizer = BertTokenizer.from_pretrained(model_name)

def prepare_input(texts):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True,
                            max_length = 50,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'tf', 
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return tf.concat(input_ids, axis=0).numpy(), tf.concat(attention_masks, axis=0).numpy()

def cross_val_score(model, X, y, bs, cv=5):
    kf = StratifiedKFold(n_splits=cv)
    scores = []

    for train_index, val_index in kf.split(X[0], y):
        X_train, X_val = (X[0][train_index], X[1][train_index]), (X[0][val_index], X[1][val_index])
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train, epochs=3, batch_size=bs, validation_data=(X_val, y_val), verbose=1)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        scores.append(val_acc)
        print(f"Validation Accuracy: {val_acc}")

    return np.mean(scores)

X_train = prepare_input(train_data['premise'].values + ' ' + train_data['hypothesis'].values)
y_train = train_data['label'].values

X_test = prepare_input(test_data['premise'].values + ' ' + test_data['hypothesis'].values)
y_test = test_data['label'].values

print('Data Loaded')

def main_model_1(i,j,bs, lr):
    bert_model = TFBertModel.from_pretrained(model_name)
    input_ids = tf.keras.Input(shape=(50,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(50,), dtype=tf.int32, name="attention_mask")
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    dropout = tf.keras.layers.Dropout(j)(bert_output[0][:, 0, :])
    hidden_layer1 = tf.keras.layers.Dense(i, activation='relu')(dropout)
    dropout2 = tf.keras.layers.Dropout(j)(hidden_layer1)
    output = tf.keras.layers.Dense(3, activation='softmax')(dropout2)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    mean_accuracy = cross_val_score(model, X_train, y_train, bs, cv=5)
    print(f"Mean Validation Accuracy: {mean_accuracy}")

    model.fit(X_train, y_train, epochs=3, batch_size=bs)

    y_pred = np.argmax(model.predict(X_test), axis=1)

    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy for layer {i} with droprate {j}, bs {bs} lr {lr}: {accuracy}")

def main_model_2(i,j,k,bs, lr):
    bert_model = TFBertModel.from_pretrained(model_name)
    input_ids = tf.keras.Input(shape=(50,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(50,), dtype=tf.int32, name="attention_mask")
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    dropout = tf.keras.layers.Dropout(k)(bert_output[0][:, 0, :])
    hidden_layer1 = tf.keras.layers.Dense(i, activation='relu')(dropout)
    dropout2 = tf.keras.layers.Dropout(k)(hidden_layer1)
    hidden_layer2 = tf.keras.layers.Dense(j, activation='relu')(dropout2)
    dropout3 = tf.keras.layers.Dropout(k)(hidden_layer2)
    output = tf.keras.layers.Dense(3, activation='softmax')(dropout3)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    mean_accuracy = cross_val_score(model, X_train, y_train, bs, cv=5)
    print(f"Mean Validation Accuracy: {mean_accuracy}")

    model.fit(X_train, y_train, epochs=3, batch_size=bs)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy for layer {i}-{j} with droprate {k} bs {bs}, lr {lr}: {accuracy}")
    
print('Start Training single layer')
for bs, lr in zip(*([32, 64, 32], [2e-5, 2e-3, 2e-3])):
    for i in [32, 64, 128, 256]:
        for j in [0.0, 0.2, 0.4]:
            main_model_1(i, j, bs, lr)

print('Start Training two layer')
for bs, lr in zip(*([32, 64, 32], [2e-5, 2e-3, 2e-3])):
    for i,j in zip(*([64,128,256], [32,64,128])):
        for k in [0.0, 0.2, 0.4]:
            main_model_2(i, j, k, bs, lr)
    
print('FINISHED')
