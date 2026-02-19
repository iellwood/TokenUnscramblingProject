import os
from datasets import load_dataset
import tokenizers
from tokenizers import BertWordPieceTokenizer

# Define the folder to save the dataset
# output_folder = "iwlst_data"
#
# # Create the folder if it doesn't exist
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# Load the iwslt2017 German-English dataset
dataset = load_dataset("iwslt2017", "iwslt2017-de-en", trust_remote_code=True)

# # Save the dataset to the specified folder
# for split in dataset.keys():
#     dataset[split].to_csv(os.path.join(output_folder, f"{split}.csv"))
#
# print(f"Dataset has been saved in the folder: {output_folder}")

# SAVE THE GERMAN SENTENCES

n = len(dataset['train'])
print('Size of training dataset =', n, 'sentences.')
strings = []
for i in range(n):
    s = dataset['train'][i]['translation']['de']
    strings.append(s + '\r')

# Concatenate all the strings into one large string
all_german_strings = ''.join(strings)  # The ''.join() method ensures there's no space between strings.

# Write the large string to a text file
with open('../iwslt_data/germantext.txt', 'w', encoding='utf-8') as file:
    file.write(all_german_strings)

# SAVE THE ENGLISH SENTENCES

strings = []
for i in range(n):
    s = dataset['train'][i]['translation']['en']
    strings.append(s + '\r')

# Concatenate all the strings into one large string
all_english_strings = ''.join(strings)  # The ''.join() method ensures there's no space between strings.

# Write the large string to a text file
with open('../iwslt_data/englishtext.txt', 'w', encoding='utf-8') as file:
    file.write(all_english_strings)

# SAVE THE TEST DATASETS

# SAVE THE GERMAN SENTENCES

n = len(dataset['test'])
strings = []
for i in range(n):
    s = dataset['test'][i]['translation']['de']
    strings.append(s + '\r')

# Concatenate all the strings into one large string
all_german_strings = ''.join(strings)  # The ''.join() method ensures there's no space between strings.

# Write the large string to a text file
with open('../iwslt_data/germantext_TEST.txt', 'w', encoding='utf-8') as file:
    file.write(all_german_strings)

# SAVE THE ENGLISH SENTENCES

strings = []
for i in range(n):
    s = dataset['test'][i]['translation']['en']
    strings.append(s + '\r')

# Concatenate all the strings into one large string
all_english_strings = ''.join(strings)  # The ''.join() method ensures there's no space between strings.

# Write the large string to a text file
with open('../iwslt_data/englishtext_TEST.txt', 'w', encoding='utf-8') as file:
    file.write(all_english_strings)