from html_parse import get_definitions
from PoS import get_pos
from helper import filter_by_pos
from WSD import  sentence2vec, cos_similarity


def main(sentence: str, target: int):

    pos = get_pos(sentence, target)
    word = sentence.split(" ")[target]

    entries:dict = get_definitions(word)
    entries:list[tuple] = filter_by_pos(entries, pos) 
    
    all = [sentence]
    all.extend([a[1] for a in entries])

    embeddings = sentence2vec(all)
    embed_context = embeddings[0]
    embed_entries = embeddings[1:]

    scores = cos_similarity(embed_context, embed_entries)
    
    rank = []
    for entry, score in zip([e[0] for e in entries], scores):
        rank.append([entry,float(score)])
    
    # Sort by score (highest first)
    rank = sorted(rank, key=lambda x: x[-1], reverse=True)

    with open("./results.html", "w") as file:
        for w in rank:
            file.write(f"{w[0]}")
            print(w[0], "score: ", w[1])

if __name__=="__main__":
    sentence = "To get a mortgage I went to the bank"
    target = 8
    main(sentence, target)
