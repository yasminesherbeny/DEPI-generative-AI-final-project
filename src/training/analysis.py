import matplotlib.pyplot as plt


def analyze_raw_lengths(dataset, tokenizer):
    lengths = [len(tokenizer(x["prompt"])["input_ids"]) for x in dataset]

    print("Raw Token Lengths:")
    print("Max:", max(lengths))
    print("Min:", min(lengths))

    plt.hist(lengths, bins=50)
    plt.title("Original Token Length Distribution")
    plt.show()


def analyze_tokenized_lengths(tokenized_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_dataset]

    print("Tokenized Lengths:")
    print("Max:", max(lengths))
    print("Min:", min(lengths))