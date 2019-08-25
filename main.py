import sys
sys.path.insert(0, "../")
import json
import DCA.dataset as D
import argparse
import DCA.utils as utils
from pprint import pprint
import torch
import pickle
from DCA.ed_ranker import EDRanker
import csv
import time

import numpy as np



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# general args
parser.add_argument('--device', type=int,
                    help="GPU device number",
                    default=0)
parser.add_argument("--mode", type=str,
                    help="train or eval",
                    default='train')
parser.add_argument("--order", type=str,
                    help="offset or random or size",
                    default='offset')
parser.add_argument("--model_path", type=str,
                    help="model path to save/load",
                    default='Model/')
parser.add_argument("--output_path", type=str,
                    help="output path to save/load",
                    default='Output1/')
parser.add_argument("--method", type=str,
                    help="training method, Supervised Learning or Reinforcement Learning",
                    default='SL')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument("--dropout_rate", type=float,
                    help="dropout rate for ranker model",
                    default=0.2)
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument("--order_learning", type=str2bool, nargs='?', default='n', const=True,
                    help="Activate order learning mode.")
parser.add_argument("--use_local_only", type=str2bool, nargs='?', default='n', const=True,
                    help="Activate local_only mode.")
parser.add_argument("--sort", type=str,
                    help="heuristic order, local similarity or topic similarity",
                    default='topic')

# args for preranking (i.e. 2-step candidate selection)
parser.add_argument("--n_cands_before_rank", type=int,
                    help="number of candidates",
                    default=50)
parser.add_argument("--prerank_ctx_window", type=int,
                    help="size of context window for the preranking model",
                    default=50)
parser.add_argument("--keep_p_e_m", type=int,
                    help="number of top candidates to keep w.r.t p(e|m)",
                    default=4)
parser.add_argument("--keep_ctx_ent", type=int,
                    help="number of top candidates to keep w.r.t using context",
                    default=4)

# args for local model
parser.add_argument("--ctx_window", type=int,
                    help="size of context window for the local model",
                    default=100)
parser.add_argument("--tok_top_n", type=int,
                    help="number of top contextual words for the local model",
                    default=50)
parser.add_argument("--tok_top_n4ment", type=int,
                    help="number of top previous disambiguated mentions for the whole model",
                    default=7)
parser.add_argument("--tok_top_n4ent", type=int,
                    help="number of top knowledge entities for the whole model",
                    default=7)
parser.add_argument("--tok_top_n4word", type=int,
                    help="number of top knowledge words for the whole model",
                    default=50)
parser.add_argument("--tok_top_n4inlink", type=int,
                    help="number of top inlinked entities for the whole model",
                    default=100)

# args for global model
parser.add_argument("--hid_dims", type=int,
                    help="number of hidden neurons",
                    default=100)

# args for training
parser.add_argument("--n_epochs", type=int,
                    help="max number of epochs",
                    default=500)
parser.add_argument("--dev_f1_change_lr", type=float,
                    help="dev f1 to change learning rate",
                    default=0.928)
parser.add_argument("--dev_f1_start_order_learning", type=float,
                    help="dev f1 to start order learning",
                    default=0.92)
parser.add_argument("--n_not_inc", type=int,
                    help="number of evals after dev f1 not increase",
                    default=20)
parser.add_argument("--eval_after_n_epochs", type=int,
                    help="number of epochs to eval",
                    default=5)
parser.add_argument("--learning_rate", type=float,
                    help="learning rate",
                    default=2e-4)
parser.add_argument("--margin", type=float,
                    help="margin",
                    default=0.01)

parser.add_argument('--seq_len', type=int, default=0,
                    help='sequence length during training')

parser.add_argument('--dca_method', type=int, default=0,
                    help='dca select method, 0: attention topk, 1: attention all, 2: average')

parser.add_argument('--isDynamic', type=int, default=0,
                    help='0: coherence+DCA, 1: coherence, 2: local model')

parser.add_argument('--one_entity_once', type=int, default=0,
                    help='')

parser.add_argument("--use_early_stop", type=str2bool, nargs='?', default='n', const=True,
                    help="")

args = parser.parse_args()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)

torch.manual_seed(args.seed)    # set random seed for cpu
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)   # set random seed for present GPU

datadir = '../data/generated/test_train_data'
conll_path = '../data/basic_data/test_datasets'
person_path = '../data/basic_data/p_e_m_data/persons.txt'
voca_emb_dir = "../data/generated/embeddings/word_ent_embs/"
ent_inlinks_path = "../data/entityid_dictid_inlinks_uniq.pkl"

timestr = time.strftime("%Y%m%d-%H%M%S")

#F1_CSV_Path = args.output_path + args.method + "_" + args.order + "_" + "f1.csv"

F1_CSV_Path = args.output_path + args.method + "_" + args.order + "_" + str(args.tok_top_n) + "-" \
              + str(args.tok_top_n4ent) + "-" + str(args.tok_top_n4word) + "-" + str(args.tok_top_n4inlink) + "_" \
              + timestr + "_" + str(args.order_learning) + "_" + args.sort + "_" + str(args.seq_len) + str(args.isDynamic) + str(args.dca_method) + str(args.one_entity_once) + "f1.csv"


if __name__ == "__main__":
    print('load conll at', datadir)
    conll = D.CoNLLDataset(datadir, conll_path, person_path, args.order, args.method)

    print('create model')
    word_voca, word_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.word',
                                                      voca_emb_dir + 'word_embeddings.npy')

    entity_voca, entity_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.entity',
                                                          voca_emb_dir + 'entity_embeddings.npy')

    with open(ent_inlinks_path, 'rb') as f_pkl:
        ent_inlinks_dict = pickle.load(f_pkl)

    config = {'hid_dims': args.hid_dims,
              'emb_dims': entity_embeddings.shape[1],
              'freeze_embs': True,
              'tok_top_n': args.tok_top_n,
              'tok_top_n4ment': args.tok_top_n4ment,
              'tok_top_n4ent': args.tok_top_n4ent,
              'tok_top_n4word': args.tok_top_n4word,
              'tok_top_n4inlink': args.tok_top_n4inlink,
              'margin': args.margin,
              'word_voca': word_voca,
              'entity_voca': entity_voca,
              'word_embeddings': word_embeddings,
              'entity_embeddings': entity_embeddings,
              'entity_inlinks': ent_inlinks_dict,
              'dr': args.dropout_rate,
              'gamma': args.gamma,
              'order_learning': args.order_learning,
              'dca_method' : args.dca_method,
              'f1_csv_path': F1_CSV_Path,
              'seq_len': args.seq_len,
              'isDynamic' : args.isDynamic,
              'one_entity_once': args.one_entity_once,
              'args': args}

    # pprint(config)
    ranker = EDRanker(config=config)

    dev_datasets = [
                    # ('aida-train', conll.train),
                    ('aida-A', conll.testA),
                    ('aida-B', conll.testB),
                    ('msnbc', conll.msnbc),
                    ('aquaint', conll.aquaint),
                    ('ace2004', conll.ace2004),
                    ('clueweb', conll.clueweb),
                    ('wikipedia', conll.wikipedia)
                ]

    with open(F1_CSV_Path, 'w') as f_csv_f1:
        f1_csv_writer = csv.writer(f_csv_f1)
        f1_csv_writer.writerow(['dataset', 'epoch', 'dynamic', 'F1 Score'])

    if args.mode == 'train':
        print('training...')
        config = {'lr': args.learning_rate, 'n_epochs': args.n_epochs, 'isDynamic':args.isDynamic, 'use_early_stop' : args.use_early_stop,}
        # pprint(config)
        ranker.train(conll.train, dev_datasets, config)

    elif args.mode == 'eval':
        org_dev_datasets = dev_datasets  # + [('aida-train', conll.train)]
        dev_datasets = []
        for dname, data in org_dev_datasets:
            dev_datasets.append((dname, ranker.get_data_items(data, predict=True)))
            print(dname, '#dev docs', len(dev_datasets[-1][1]))

        for di, (dname, data) in enumerate(dev_datasets):
            predictions = ranker.predict(data)
            print(dname, utils.tokgreen('micro F1: ' + str(D.eval(org_dev_datasets[di][1], predictions))))

