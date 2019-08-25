from DCA.vocabulary import Vocabulary
import numpy as np

stopword_input = "../data/stopwords-multi.txt"

with open(stopword_input, 'r') as f_in:
    stop_word = f_in.readlines()

stop_words = [z.strip() for z in stop_word]

symbol_input = "../data/symbols.txt"

with open(symbol_input, 'r') as f_sy:
    symbol = f_sy.readlines()

symbols = [s.strip() for s in symbol]


def is_important_word(s):
    """
    an important word is not a stopword, a number, or len == 1
    """
    try:
        if len(s) <= 1 or s.lower() in stop_words or s.lower() in symbols:
            return False
        float(s)
        return False
    except:
        return True

############################ process list of lists ###################


def flatten_list_of_lists(list_of_lists):
    """
    making inputs to torch.nn.EmbeddingBag
    """
    list_of_lists = [[]] + list_of_lists
    offsets = np.cumsum([len(x) for x in list_of_lists])[:-1]
    flatten = sum(list_of_lists[1:], [])
    return flatten, offsets


def load_voca_embs(voca_path, embs_path):
    voca = Vocabulary.load(voca_path)
    embs = np.load(embs_path)

    # check if sizes are matched
    if embs.shape[0] == voca.size() - 1:
        unk_emb = np.mean(embs, axis=0, keepdims=True)
        embs = np.append(embs, unk_emb, axis=0)
    elif embs.shape[0] != voca.size():
        print(embs.shape, voca.size())
        raise Exception("embeddings and vocabulary have differnt number of items ")

    return voca, embs


def make_equal_len(lists, fill_in=0, to_right=True):
    lens = [len(l) for l in lists]
    max_len = max(1, max(lens))
    if to_right:
        eq_lists = [l + [fill_in] * (max_len - len(l)) for l in lists]
        mask = [[1.] * l + [0.] * (max_len - l) for l in lens]
    else:
        eq_lists = [[fill_in] * (max_len - len(l)) + l for l in lists]
        mask = [[0.] * (max_len - l) + [1.] * l for l in lens]
    return eq_lists, mask

############################### coloring ###########################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def tokgreen(s):
    return bcolors.OKGREEN + s + bcolors.ENDC


def tfail(s):
    return bcolors.FAIL + s + bcolors.ENDC


def tokblue(s):
    return bcolors.OKBLUE + s + bcolors.ENDC