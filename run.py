import torch

# This script is for "deployment". It loads a pre-trained model and uses it.

# --- Vocabulary and Encoding ---
# We need the exact same vocabulary and encoding functions as in the training script.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# --- The Model Architecture ---
# We need the exact same model class definition.
class TinyLLM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super(TinyLLM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

# --- Text Generation Function ---
def generate(model, start_string='The ', length=500):
    model.eval()
    start_tensor = torch.tensor(encode(start_string), dtype=torch.long).unsqueeze(0)
    hidden = model.init_hidden(1)
    generated_text = start_string
    for _ in range(length):
        output, hidden = model(start_tensor, hidden)
        probabilities = torch.nn.functional.softmax(output[0, -1, :], dim=0)
        next_char_idx = torch.multinomial(probabilities, 1).item()
        generated_char = int_to_string[next_char_idx]
        generated_text += generated_char
        start_tensor = torch.tensor([[next_char_idx]], dtype=torch.long)
    return generated_text

# --- Load the Trained Model ---
# Define the model with the same hyperparameters used during training.
embedding_dim = 128
hidden_dim = 256
n_layers = 2
model = TinyLLM(vocab_size, embedding_dim, hidden_dim, n_layers)

# Load the saved weights (the "state dictionary").
model.load_state_dict(torch.load('tinyllm.pth'))
print("Model loaded successfully from tinyllm.pth")

# --- Generate Text ---
print("\n--- Generating Text from Loaded Model ---")
prompt = "He looked at me with a "
generated_text = generate(model, start_string=prompt, length=1000)
print(generated_text)