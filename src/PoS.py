from nltk import pos_tag, word_tokenize
from maps import PENN_TO_SIMPLE

#nltk.download("averaged_perceptron_tagger_eng")
#nltk.download("punkt_tab")

def get_pos(sentence:str, target:int) -> list[list[str]]:
    """
    Returns the PoS tag of the input
    Args:
        - sentence(str): sentence to tag
    """
    tokens = word_tokenize(sentence)
    pos = pos_tag(tokens)[target][1]

    return PENN_TO_SIMPLE[pos]


if __name__ == "__main__":
    stnc = "After the duck ducked under the lead pipe, the soldiers decided to deset their post, deserting nothing behind, while reading the book on bass fishing"
    print(get_pos(stnc, 2))
