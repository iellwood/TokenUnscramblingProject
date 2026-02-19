from LearnedTransformerAttentionProject.DataImportUtils.loadtextfile import load_file_lines
import tokenizers
import numpy as np

def make_tokenized_dataset(tokenizer_path, dataset_path, tokenized_dataset_path, max_length=None):
    lines = load_file_lines(dataset_path)
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)


    if max_length is None:
        ids_list = []
        max_length = 0
        for line in lines:
            ids = tokenizer.encode(line).ids
            ids_list.append(ids)
            if len(ids) > max_length:
                max_length = len(ids)
    else:
        ids_list = []
        for line in lines:
            ids = tokenizer.encode(line).ids
            ids_list.append(ids)

    id_array = np.zeros(shape=(len(ids_list), max_length), dtype=np.int64)

    for i, ids in enumerate(ids_list):
        for j, id in enumerate(ids):
            id_array[i, j] = id + 2  # Add 2 to make 0 and 1 special tokens representing the start and end of a sequence

    np.save(tokenized_dataset_path, np.transpose(id_array, axes=(1, 0)))

# make_tokenized_dataset("bytelevel_tokenizer_ENGLISH.json", "../iwslt_data/englishtext.txt", "../iwslt_data/english_ids")
# make_tokenized_dataset("bytelevel_tokenizer_GERMAN.json", "../iwslt_data/germantext.txt", "../iwslt_data/german_ids")

make_tokenized_dataset("bytelevel_tokenizer_ENGLISH.json", "../iwslt_data/englishtext_TEST.txt", "../iwslt_data/english_ids_TEST", max_length=227)
make_tokenized_dataset("bytelevel_tokenizer_GERMAN.json", "../iwslt_data/germantext_TEST.txt", "../iwslt_data/german_ids_TEST", max_length=203)