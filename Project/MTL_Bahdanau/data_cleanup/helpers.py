import pickle


def loadPickle(sourceName: str, targetName: str):
    with open(sourceName + ".pkl", "rb") as f:
        source_tok = pickle.load(f)

    with open(targetName + ".pkl", "rb") as f:
        target_tok = pickle.load(f)

    return source_tok, target_tok


def savePickle(sourceName: str, sourceObj, targetName: str, targetObj):
    with open(sourceName + ".pkl", "wb") as f:
        pickle.dump(sourceObj, f)

    with open(targetName + ".pkl", "wb") as f:
        pickle.dump(targetObj, f)
