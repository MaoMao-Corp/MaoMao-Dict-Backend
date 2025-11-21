from WSD import WordSenseDisambiguator


def main():
    
    wsd = WordSenseDisambiguator(use_fp16=True)
    while True:
        sentence = input("sentence: ")
        index = int(input("index: "))
        
        wsd.disambiguate(sentence, index, verbose=True)
        
        input("Press enter to return")

if __name__=="__main__":
    main()
