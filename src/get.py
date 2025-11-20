from pystardict import Dictionary
from maps import POS_DOBLE


a = input("word: ")
d = Dictionary("./dict/stardict")

print(d[a])
