from collections import Counter
import random
class Corpus(object):
    def __init__(self, entries: list=None):
        self.entries = []
        if entries is not None:
            for e in entries:
                self.add_entry(e["description"], e["label"])
        
    def __len__(self):
        return len(self.entries)

    def __str__(self):
        return '\n'.join(map(repr, self.entries))

    def __getitem__(self, item: int): #-> Corpus:
        return Corpus(entries=self.entries[item])
    
    def add_entry(self, description: str, label: str) -> None:
        self.entries.append({"description": description, "label": label})

    def shuffle(self) -> None:
        random.shuffle(self.entries)

    def add_predictions(self, predictions: list) -> None:
        for entry, p in zip(self.entries, predictions):
            entry["label"] = p
    
    @property
    def descriptions(self) -> list:
        return [entry["description"] for entry in self.entries]

    @property
    def status_labels(self) -> list:
        return [entry["label"] for entry in self.entries]

    @property
    def statuses(self) -> list:
        statuses = list(set([entry["label"] for entry in self.entries]))
        statuses.sort()
        return statuses

    @property
    def class_distribution(self) -> Counter:
        return Counter(self.status_labels)


    def save(self, filepath: str) -> None:
        with open(filepath, "w") as o:
            for e in self.entries:
                output_items = [e["label"], e["description"]]
                output = ": ".join(output_items)
                o.write(output)
                o.write("\n")

def read_data(filepath: str, predict: bool=False) -> list:
    corpus = Corpus()
    with open(filepath, "r") as f:
        for line in f:
            if predict:
                label = ""
                description = line.strip()
            else:
                items = line.strip().split(":")
                label = items[0]
                description = line[len(label)+2:].strip()
            corpus.add_entry(description, label)
    return corpus
