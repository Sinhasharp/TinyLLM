import torch

# --- Data Loading and Preparation ---

# Open the text file and read it all into a single string.
# Make sure you have the 'input.txt' file in the same directory.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get the set of all unique characters in the text.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create the character-to-integer and integer-to-character mappings.
# These are our "encoder" and "decoder".
string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([int_to_string[i] for i in l]) # decoder: take a list of integers, output a string

# Let's test our encoder/decoder.
encoded_text = encode("hello world")
decoded_text = decode(encoded_text)

print("--- Data Preparation Check ---")
print(f"Length of dataset in characters: {len(text)}")
print(f"Vocabulary size (unique characters): {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")
print(f"--- Encoder/Decoder Test ---")
print(f"Original: 'hello world'")
print(f"Encoded: {encoded_text}")
print(f"Decoded: {decoded_text}")

# (Keep all the code from Step 2 above this line)

# --- The Model Architecture ---

class TinyLLM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super(TinyLLM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 1. Embedding Layer: Converts character integers into dense vectors.
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # 2. LSTM Layer: The core of the model for processing sequences.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        
        # 3. Fully Connected Layer: Maps the LSTM output to our vocabulary.
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # The forward pass defines how data flows through the layers.
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        # Creates the initial hidden state and cell state for the LSTM.
        # These are essentially the model's "short-term memory" at the start.
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

# --- Model Hyperparameters ---
# These are the settings for our model's size and complexity.
# We'll keep them small to run on a CPU.
embedding_dim = 128   # Size of the character embedding vectors
hidden_dim = 256      # Number of neurons in the LSTM hidden layer
n_layers = 2          # Number of layers in the LSTM

# Create an instance of our model
model = TinyLLM(vocab_size, embedding_dim, hidden_dim, n_layers)

# --- Sanity Check ---
# Let's print the model to see its structure.
print("\n--- Model Architecture ---")
print(model)

# (Keep all the code from the previous steps above this line)

# --- Preparing Data for Training ---

# First, convert our entire text into a sequence of integers.
data = torch.tensor(encode(text), dtype=torch.long)

def get_batch(batch_size, sequence_length):
    # This function creates a batch of input sequences (x) and target sequences (y).
    # It randomly selects starting points in the data.
    starts = torch.randint(0, len(data) - sequence_length, (batch_size,))
    x = torch.stack([data[i:i+sequence_length] for i in starts])
    y = torch.stack([data[i+1:i+sequence_length+1] for i in starts])
    return x, y

# (Your TinyLLM class definition should be above this)

# --- Text Generation ---

def generate(model, start_string='The ', length=500):
    model.eval()  # Set the model to evaluation mode
    
    # Encode the starting string and convert to a tensor
    start_tensor = torch.tensor(encode(start_string), dtype=torch.long).unsqueeze(0)
    
    # Initialize the hidden state
    hidden = model.init_hidden(1) # Batch size is 1 for generation

    # Generate characters one by one
    generated_text = start_string
    for _ in range(length):
        # Get the output from the model
        output, hidden = model(start_tensor, hidden)
        
        # Get the probability distribution of the next character
        # and sample from it to get the actual next character
        probabilities = torch.nn.functional.softmax(output[0, -1, :], dim=0)
        next_char_idx = torch.multinomial(probabilities, 1).item()
        
        # Add the new character to our text and update the input for the next round
        generated_char = int_to_string[next_char_idx]
        generated_text += generated_char
        start_tensor = torch.tensor([[next_char_idx]], dtype=torch.long)
        
    model.train() # Set the model back to training mode
    return generated_text

# (The "Model Hyperparameters" section and the rest of your code follows)

# --- Training Hyperparameters and Components ---
batch_size = 64        # How many independent sequences to process in parallel.
sequence_length = 256  # The maximum context length for predictions.
learning_rate = 1e-3   # A small number that controls how much we adjust the model.

# The Optimizer: Adam is a popular and effective choice. It takes the model's
# parameters and the learning rate to perform the updates.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# The Loss Function: It measures the difference between the model's prediction
# and the actual next character. CrossEntropyLoss is standard for this type of problem.
criterion = torch.nn.CrossEntropyLoss()

# --- The Training Loop ---
print("\n--- Starting Training ---")
print("Train for a while (at least to step 500+) then press Ctrl+C to stop and generate text.")

n_steps = 10000 # Increased steps so you can train for longer if you wish
print_every = 100

try:
    for step in range(n_steps):
        hidden = model.init_hidden(batch_size)
        x, y = get_batch(batch_size, sequence_length)
        outputs, hidden = model(x, hidden)
        loss = criterion(outputs.view(-1, vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_every == 0:
            print(f"Step {step}/{n_steps}, Loss: {loss.item():.4f}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")

# (Your try...except KeyboardInterrupt block)

finally:
    print("\n--- Generating Text ---")
    generated_text = generate(model, start_string="Sherlock was a man who ", length=500)
    print(generated_text)

    # ADD THIS LINE:
    torch.save(model.state_dict(), 'tinyllm.pth')
    print("\n--- Model saved to tinyllm.pth ---")

    print("\n--- Generation Complete ---")

print("\n--- Training Check Complete ---")