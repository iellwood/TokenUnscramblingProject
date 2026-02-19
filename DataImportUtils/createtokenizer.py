import tokenizers

def make_tokenizer_from_file(filename, savefilename):
    # Initialize a tokenizer
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    tokenizer.post_processor = tokenizers.processors.ByteLevel(trim_offsets=True)

    # And then train
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=9998,
        min_frequency=2,
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train([filename], trainer=trainer)

    tokenizer.save(savefilename, pretty=True)

make_tokenizer_from_file("../iwslt_data/englishtext.txt", "../Tokenizers/bytelevel_tokenizer_ENGLISH.json")
make_tokenizer_from_file("../iwslt_data/germantext.txt", "../Tokenizers/bytelevel_tokenizer_GERMAN.json")

#
# # Function to load a text file and return a list of strings (one per line)
# def load_file_lines(file_path):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()  # Reads all lines into a list
#         return [line.strip() for line in lines]  # Removes extra whitespace/newlines
#     except FileNotFoundError:
#         print(f"Error: The file at '{file_path}' was not found.")
#         return []
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return []
#
# file_path = "../iwslt_data/englishtext.txt"
# lines = load_file_lines(file_path)
#
#
# max_length = 0
# for line in lines:
#     t = tokenizer.encode(line)
#
#     if len(t.ids) > max_length:
#         max_length = len(t.ids)
#         print(max_length)
