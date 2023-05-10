import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


train_df = pd.read_csv('./Datasets/train.csv')

train_data, test_data = train_test_split(train_df, test_size=0.2, random_state=141)

model_name = 'distilbert-base-multilingual-cased'

tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def prepare_input(texts):
    """
    Encodes a list of texts into input_ids and attention_masks for a transformer model.

    Args:
    - texts: a list of strings to be encoded.

    Returns:
    - input_ids: a numpy array of shape (batch_size, max_length) containing the encoded input sequences.
    - attention_masks: a numpy array of shape (batch_size, max_length) containing the attention masks.

    Note:
    The tokenizer object used for encoding is assumed to be already defined in the global scope.
    """
    # Initialize empty lists to store input_ids and attention_masks.
    input_ids = []
    attention_masks = []
    
    # Loop over each text and encode it.
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True,  # Add [CLS] and [SEP] tokens.
                            max_length = 50,  # Truncate or pad the sequence to a fixed length of 50.
                            pad_to_max_length = True,  # Pad the sequence with zeros if it is shorter than max_length.
                            return_attention_mask = True,  # Return the attention mask to differentiate between padding and non-padding tokens.
                            return_tensors = 'tf',  # Return the encoded inputs as TensorFlow tensors.
                       )
        
        # Append the input_ids and attention_masks to the respective lists.
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return tf.concat(input_ids, axis=0).numpy(), tf.concat(attention_masks, axis=0).numpy()

def cross_val_score(model, X, y, bs, cv=5):
    """
    Computes the cross-validated accuracy score

    Args:
    - model: a compiled Keras model to be evaluated.
    - X: a tuple of numpy arrays or tensors representing the input data.
    - y: a numpy array or tensor representing the target data.
    - bs: an integer specifying the batch size for training and evaluation.
    - cv: an integer specifying the number of folds for cross-validation.

    Returns:
    - mean_accuracy: a float representing the mean cross-validated accuracy score.

    Note:
    The function assumes that the input data and target data have the same number of samples.
    """

    # Initialize a StratifiedKFold object with cv splits.
    kf = StratifiedKFold(n_splits=cv)

    # Initialize an empty list to store the validation accuracy scores.
    scores = []

    # Loop over the cross-validation splits.
    for train_index, val_index in kf.split(X[0], y):
        # Split the input and target data into training and validation sets.
        X_train, X_val = (X[0][train_index], X[1][train_index]), (X[0][val_index], X[1][val_index])
        y_train, y_val = y[train_index], y[val_index]

        # Fit the model on the training set for 3 epochs.
        model.fit(X_train, y_train, epochs=3, batch_size=bs, validation_data=(X_val, y_val), verbose=1)

        # Evaluate the model on the validation set and append the accuracy score to the list.
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        scores.append(val_acc)

        # Print the validation accuracy for the current fold.
        print(f"Validation Accuracy: {val_acc}")

    # Compute the mean cross-validated accuracy score.
    mean_accuracy = np.mean(scores)

    return mean_accuracy

X_train = prepare_input(train_data['premise'].values + ' ' + train_data['hypothesis'].values)
y_train = train_data['label'].values

X_test = prepare_input(test_data['premise'].values + ' ' + test_data['hypothesis'].values)
y_test = test_data['label'].values

print('Data Loaded')

def main_model_1(i,j,bs, lr):
    """
    This function creates and trains a model with single hidden layer

    Args:
    i (int): The number of units in the first dense hidden layer.
    j (float): The dropout rate for the Dropout layers.
    bs (int): The batch size for training and cross-validation.
    lr (float): The learning rate for the Adam optimizer.

    Returns:
    None. The function prints mean validation and test accuracies after training and testing the model.
    """
    
    bert_model = TFDistilBertModel.from_pretrained(model_name)
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
    """
    This function creates and trains a model with 2 hidden layers

    Args:
    i (int): The number of units in the first dense hidden layer.
    j (int): The number of units in the second dense hidden layer.
    k (float): The dropout rate for the Dropout layers.
    bs (int): The batch size for training and cross-validation.
    lr (float): The learning rate for the Adam optimizer.

    Returns:
    None. The function prints mean validation and test accuracies after training and testing the model.
    """
    
    bert_model = TFDistilBertModel.from_pretrained(model_name)
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
for bs, lr in zip(*([32, 64], [2e-5, 2e-5])):
    for i in [32, 64, 128, 256]:
        for j in [0.0, 0.2, 0.4]:
            main_model_1(i, j, bs, lr)

print('Start Training two layer')
for bs, lr in zip(*([32, 64], [2e-5, 2e-5])):
    for i,j in zip(*([64,128,256], [32,64,128])):
        for k in [0.0, 0.2, 0.4]:
            main_model_2(i, j, k, bs, lr)
    
print('FINISHED')