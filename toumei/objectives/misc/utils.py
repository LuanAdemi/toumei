def convertUnitString(x: str):
    identifiers = x.split(":")
    indices = map(int, identifiers[1:])
    return tuple([identifiers[0]] + list(indices))
