from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import streamlit as st

# Define your symptoms and potential problems
symptoms = [
    "L'ordinateur ne démarre pas",
    "L'écran est noir",
    "L'ordinateur est lent",
    "L'ordinateur surchauffe",
    "Le réseau ne fonctionne pas",
    "Les périphériques USB ne sont pas reconnus",
    "L'écran clignote",
    "L'ordinateur émet des bip au démarrage",
    "Les applications se ferment de manière inattendue",
    "L'ordinateur affiche des messages d'erreur",
    "Les touches du clavier ne fonctionnent pas",
    "Les ventilateurs font un bruit anormal"
]

potential_problems = [
    "Problème d'alimentation, carte mère défectueuse",
    "Problème d'écran, carte graphique défectueuse",
    "Problème de mémoire, disque dur plein",
    "Problème de ventilation, accumulation de poussière",
    "Problème de connexion, carte réseau défectueuse",
    "Problème de pilotes, ports USB défectueux",
    "Problème de carte graphique, pilotes graphiques défectueux",
    "Problème de RAM, carte mère défectueuse",
    "Problème de mémoire, virus informatique",
    "Problème de système d'exploitation, fichiers système corrompus",
    "Problème de clavier, pilotes de clavier défectueux",
    "Problème de ventilateur, accumulation de poussière"
]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(symptoms, potential_problems, test_size=0.2, random_state=42)

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(potential_problems))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input text and convert to PyTorch tensors
train_input_ids = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")['input_ids']
train_labels = torch.tensor([potential_problems.index(problem) for problem in y_train])

# Create a PyTorch dataset
train_dataset = TensorDataset(train_input_ids, train_labels)

# Define a DataLoader for the training dataset
train_dataloader = DataLoader(train_dataset, batch_size=4)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fine-tune the model
model.train()
for epoch in range(5):  # You can adjust the number of epochs
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, labels = batch
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')

# Calculate accuracy on the testing set
test_input_ids = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")['input_ids']
test_labels = torch.tensor([potential_problems.index(problem) for problem in y_test])

test_dataset = TensorDataset(test_input_ids, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=4)

model.eval()
predictions = []
true_labels = []
for batch in test_dataloader:
    with torch.no_grad():
        input_ids, labels = batch
        outputs = model(input_ids)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend(labels.tolist())

# Convert predicted labels back to problem descriptions
predicted_problems = [potential_problems[prediction] for prediction in predictions]
true_problems = [potential_problems[label] for label in true_labels]

# Calculate accuracy
accuracy = accuracy_score(true_problems, predicted_problems)

# Streamlit app
st.title('Diagnostique de panne')

# Input text box for user to enter symptom
symptom = st.text_input('Entrer la panne rencontrée:', '')

# Button to trigger prediction
if st.button('Diagnostique'):
    # Tokenize the input symptom
    input_ids = tokenizer(symptom, padding=True, truncation=True, return_tensors="pt")['input_ids']

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    # Display predicted problem
    predicted_problem = potential_problems[prediction]
    st.write(f"Le problème prédit: {predicted_problem}")

st.write("created by fatima sekri")

    
