import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,TFAutoModel
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np



# Load and preprocess the data
data = pd.read_csv('df_total.csv')
labels = data.Type.values
sentences = data.news.tolist()

# Split data into train and test sets
train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Split train data into train and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_sentences, train_labels, test_size=0.2, random_state=42)

# One-hot encode the labels
encoder = LabelEncoder()
encoder.fit(labels)
train_labels = to_categorical(encoder.transform(train_labels))
val_labels = to_categorical(encoder.transform(val_labels))
test_labels = to_categorical(encoder.transform(test_labels))

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
# Pad sequences to a fixed length of 512
train_encodings = tokenizer(train_sentences, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_sentences, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_sentences, truncation=True, padding=True, max_length=512)

# Convert to numpy arrays
train_inputs = [pad_sequences(train_encodings['input_ids'], maxlen=512),
                pad_sequences(train_encodings['attention_mask'], maxlen=512)]
val_inputs = [pad_sequences(val_encodings['input_ids'], maxlen=512),
              pad_sequences(val_encodings['attention_mask'], maxlen=512)]
test_inputs = [pad_sequences(test_encodings['input_ids'], maxlen=512),
               pad_sequences(test_encodings['attention_mask'], maxlen=512)]
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

# Define model architecture
input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
input_attention = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')

# bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_model = TFAutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
bert_outputs = bert_model.bert(input_ids, attention_mask=input_attention)
last_hidden_state = bert_outputs.last_hidden_state

output = tf.keras.layers.Dense(7, activation='softmax', name='output')(last_hidden_state[:, 0, :])

model = tf.keras.models.Model(inputs=[input_ids, input_attention], outputs=output)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_inputs, train_labels, epochs=50, batch_size=16, validation_data=(val_inputs, val_labels))

# Create a new figure and axis object for training and validation accuracy plot
fig1, ax1 = plt.subplots()
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['train', 'val'], loc='upper left')

# Create a new figure and axis object for training and validation loss plot
fig2, ax2 = plt.subplots()
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['train', 'val'], loc='upper left')

# Display the plots and save as separate image files
fig1.savefig('accuracy.png')
fig2.savefig('loss.png')
plt.show()

# Get predictions for test data
test_preds = model.predict(test_inputs)
test_preds = np.argmax(test_preds, axis=1)


# Get true labels for test data
test_true = np.argmax(test_labels, axis=1)

# Calculate test accuracy
test_accuracy = np.sum(test_preds == test_true) / len(test_true)
print(f'Test accuracy: {test_accuracy:.2%}')

cm = confusion_matrix(test_true, test_preds)
fig3, ax3 = plt.subplots(figsize=(10, 10))
im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
ax3.set_title('Confusion Matrix')
fig3.colorbar(im, ax=ax3)
tick_marks = np.arange(len(encoder.classes_))
ax3.set_xticks(tick_marks)
ax3.set_yticks(tick_marks)
ax3.set_xticklabels(encoder.classes_, rotation=30)
ax3.set_yticklabels(encoder.classes_)
for i in range(len(encoder.classes_)):
    for j in range(len(encoder.classes_)):
        ax3.text(j, i, cm[i, j], ha='center', va='center', color='white')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('True')
fig3.savefig('confusion_matrix.png')
plt.show()












