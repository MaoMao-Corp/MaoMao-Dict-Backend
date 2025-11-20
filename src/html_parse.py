from pprint import pprint
import re
from bs4 import BeautifulSoup
from maps import stardict, POS_DOBLE, DICT_TO_SIMPLE, POS_SET

def is_entry(b, p, word) -> bool:
    return b and (b.get_text(strip=True)==word.capitalize() or b.get_text(strip=True).endswith(".")) and "><b>Syn." not in p.get_text()

def is_new_pos(b, word) -> bool:
    return b.get("style") and "color: #00b" in b["style"] and b.string and word.capitalize() in b.string

def add_entry(dic,pos,p):
    try:
        if pos not in dic: dic[pos] = []
        txt = clean_text(p.get_text())
        dic[pos].append({'html':p, 'definition': txt})
    except AttributeError:
        pass

def clean_text(text):
    new = re.sub(r"\[[^\]]+\]", "", text)
    return new



def get_definitions(word:str) -> dict[str, list[dict]]:
    """
    Args:
        - word: word to fetch definition of
    Returns:
        Dict of the form:
        {
         'pos' : [(p1, p1.txt), (p2, p.txt2)],
        }
    """

    html = stardict[word] 
    soup = BeautifulSoup(html, "html.parser")

    entries = {}
    
    curr_pos = "unk"
    for p in soup.find_all("p"):
        b = p.find("b", recursive=False)
        if is_entry(b, p, word):
            if is_new_pos(b, word):
                try:
                    pos = p.find("i", style=lambda v: v and "color: #a00" in v).get_text(strip=True).replace("&","&amp;")
                except Exception as e:
                    pos = "unk"
                if pos in POS_DOBLE: 
                    l_pos = POS_DOBLE[pos]
                    curr_pos = []
                    for p in l_pos:
                        if p in POS_SET: curr_pos.append(p)
                        else:curr_pos.append(DICT_TO_SIMPLE[p])

                elif pos in POS_SET: current_pos = pos

                else:   curr_pos = DICT_TO_SIMPLE[pos]
            if type(curr_pos)==list: 
                for cp in curr_pos:
                    add_entry(entries,cp, p)
            else: add_entry(entries, curr_pos, p)

    return entries

if __name__ == "__main__":
    #word = input("word: ")
    #print(get_definitions(word))
    #quit()
    count=0
    for word in stardict.keys():
        try:
            d = get_definitions(word)
            for k,v in d.items():
                count+=len(v)
        except:
            pass
    print(count)
