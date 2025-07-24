import pickle


def LoadPickle(path_eng, path_fra):
    with open(path_eng, "rb") as f:
        eng = pickle.load(f)

    with open(path_fra, "rb") as f:
        fra = pickle.load(f)

    return eng, fra
