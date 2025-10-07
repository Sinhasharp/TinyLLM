# My First Language Model: TinyLLM!

I'm so excited to share my first real machine learning project! This repository is the result of my journey into building a language model completely from scratch using Python and PyTorch. I wanted to understand how giants like ChatGPT actually work, and this project was the perfect way to learn the fundamentals without needing a supercomputer.

The model I built is a character-level Recurrent Neural Network (RNN). It's amazing! It reads a book and learns, character by character, how to write in the same style.

## How It All Works!
* I trained my model on "The Adventures of Sherlock Holmes," and it was incredible to see it learn. Here’s the process I followed:

* Data Processing: First, I had to teach the computer to read. I wrote code to scan the entire book, find every unique character, and give each one a special number. It was like creating a secret codebook for the text!

* Building the Model's "Brain": This was the most exciting part. I used PyTorch to design a simple neural network with LSTM cells. I thought of it as building a small brain with an embedding layer (to understand the characters), LSTM layers (to have memory of what it just "read"), and a final layer to make its guess for the next character.

* The Training Loop: This is where the magic happens. I fed the model small, random pieces of the book and asked it, "What character comes next?" At first, its guesses were random nonsense. But with every guess, I calculated its "error" (the loss) and used that to make tiny adjustments to its brain. I did this thousands of times, and I could actually watch it getting smarter as the loss value went down!

* Generating Text!: After some training, I could finally ask my model to write something new. I'd give it a starting phrase, and it would predict the next character, add it, and then use that new phrase to predict the next character. Seeing it generate text, even if it was gibberish, was an incredible moment.

## Getting Started

If you want to try this yourself, here’s how to get it running. I've tried to make it as simple as possible!

### Prerequisites

* Python 3.9 or higher
* pip (Python package installer)

### Installation

* #### Clone this Repository
  
* #### Activate the environment
  ```bash
  # Create the environment
  python -m venv venv

  # Activate on Windows
  .\venv\Scripts\activate

  # Activate on macOS/Linux
  source venv/bin/activate
  ```
  
* #### Install the required libraries:
  ```bash
  pip install -r requirements.txt
  ```

### Let's Make it Learn!
  * #### Add the Dataset
    You need to give the model something to read. I used a classic from Project Gutenberg.

    Link: https://www.gutenberg.org/files/1661/1661-0.txt

    Download the file, save it in the project folder, and rename it to input.txt.

  * #### Train the Model
    This is where you watch it come to life. Run the main.py script.
    ```bash
    python main.py
    ```

      * The training will start, and it's going to be slow on a normal computer, but be patient!

      * You'll see the "loss" printed every 100 steps. Watching this number go down is the best part—it's proof that it's learning!

      * Let it run for a while (at least 500-1000 steps), then press Ctrl+C to stop it.

      * When you stop it, it will automatically show you a sample of what it learned to write and save its "brain" to a file called tinyllm.pth.

  * #### Generate Text with Your Trained Model!
      Once tinyllm.pth exists, you can make the model write for you anytime without retraining!
      ```bash
      python run.py
      ```
      This script loads the brain you saved and instantly generates new text.

## My First Result!

After just 1000 training steps, the output was hilarious nonsense, but it had structure! It learned to put spaces between letters to form "words," and it used punctuation. It was trying to write like a human! For a first attempt, I was thrilled with this result.

## Project Structure

```bash
TinyLLM/
├── venv/                     # The virtual environment folder
├── main.py                   # The main script I used for training
├── run.py                    # The script to run the trained model
├── input.txt                 # The training book (you add this!)
├── tinyllm.pth               # The model's saved brain (gets created after training)
├── requirements.txt          # The Python libraries needed
└── README.md                 # This file!
```

## What I Learned on this Journey
This project was an amazing first step for me. I finally feel like I understand the basics of:

  * How text is turned into numbers for a machine (tokenization).

  * What the main parts of a neural network in PyTorch are.

  * The concept of a training loop, loss, and optimizers. It's not magic, it's just math!

  * How to save a model's progress and use it later.

This was my first major project in machine learning, and I had a blast making it. Hope you enjoy trying it out!

#### If you are at this point and if you liked my project then please leave a star.
Thankyou,
Sinhasharp
