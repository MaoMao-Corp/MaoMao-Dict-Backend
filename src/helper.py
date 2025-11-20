


def filter_by_pos(dic, pos):
    if pos not in dic:
        return [e for innerList in dic.values() for e in innerList] 
    return dic[pos]
