import os
import pyarrow as pa
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# --- Configuration (matching Node.js script) ---
# Resolve paths relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(script_dir, '../datasets/daily_dialog/default/1.0.0/1d0a58c7f2a4dab5ed9d01dbde8e55e0058e589ab81fce5c2df929ea810eabcd/daily_dialog-train.arrow')
OUTPUT_TOKENIZER_PATH = os.path.join(script_dir, 'daily_dialog_tokenizer.json')
VOCAB_SIZE = 10000  # Target vocabulary size
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

# --- Helper Function to Extract Text ---
def extract_text_from_arrow(file_path):
    """Reads an Arrow file and extracts text from the 'dialog' column."""
    print(f"Reading Arrow file: {file_path}")
    try:
        table = pa.ipc.open_file(file_path).read_all() # Correct way to read Arrow IPC file
        print(f"Loaded table with {table.num_rows} rows.")

        dialog_column = table.column('dialog')
        if dialog_column is None:
            raise ValueError("Could not find 'dialog' column in Arrow table.")

        utterances = []
        # Correctly iterate over chunks
        for dialogue_chunk in dialog_column.chunks:
            for dialogue_list in dialogue_chunk:
                 # Check if it's a valid list-like structure from Arrow
                 if dialogue_list is not None and hasattr(dialogue_list, 'as_py'):
                     py_list = dialogue_list.as_py()
                     if isinstance(py_list, list):
                         utterances.extend(item for item in py_list if isinstance(item, str)) # Add individual utterances
                     else:
                         # Find the approximate original index for warning (optional, can be complex)
                         print(f"Warning: Encountered non-list dialogue data: {type(py_list)}")
                 else:
                     print(f"Warning: Skipping invalid dialogue data: {dialogue_list}")


        print(f"Extracted {len(utterances)} utterances.")
        if not utterances:
             raise ValueError("No text data extracted from the dataset.")
        return utterances
    except Exception as e:
        print(f"Error reading or processing Arrow file: {e}")
        raise

# --- Main Training Function ---
def train_tokenizer():
    """Trains and saves a BPE tokenizer."""
    try:
        # 1. Extract Text Data
        training_texts = extract_text_from_arrow(DATASET_PATH)

        # 2. Initialize Tokenizer Model (BPE)
        # Use the standard Hugging Face tokenizers library
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        # 3. Set PreTokenizer (ByteLevel is common)
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

        # 4. Initialize Trainer
        trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

        # 5. Train Tokenizer
        print(f"Starting tokenizer training with vocab size {VOCAB_SIZE}...")
        # Use an iterator to avoid loading all text into memory at once if needed,
        # but for 87k utterances, direct list is likely fine.
        tokenizer.train_from_iterator(training_texts, trainer=trainer)
        print("Tokenizer training finished.")

        # 6. Save Tokenizer
        # Ensure the output directory exists (it should, as it's the script's dir)
        os.makedirs(os.path.dirname(OUTPUT_TOKENIZER_PATH), exist_ok=True)
        tokenizer.save(OUTPUT_TOKENIZER_PATH)
        print(f"Tokenizer saved to {OUTPUT_TOKENIZER_PATH}")

    except Exception as e:
        print(f"Error training tokenizer: {e}")
        exit(1)

# --- Run Training ---
if __name__ == "__main__":
    train_tokenizer()