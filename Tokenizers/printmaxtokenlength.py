import tokenizers


# Function to load a text file and return a list of strings (one per line)
def load_file_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()  # Reads all lines into a list
        return [line.strip() for line in lines]  # Removes extra whitespace/newlines
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def get_max_token_length(tokenizer_filename, dataset_filename):
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_filename)

    lines = load_file_lines(file_path=dataset_filename)
    max_length = 0
    for line in lines:
        t = tokenizer.encode(line)

        if len(t.ids) > max_length:
            max_length = len(t.ids)

    print(dataset_filename, 'MAX LENGTH =', max_length)


get_max_token_length("bytelevel_tokenizer_ENGLISH.json", "../iwslt_data/englishtext.txt")
get_max_token_length("bytelevel_tokenizer_GERMAN.json","../iwslt_data/germantext.txt")

