import torch
import torch.nn as nn
import io
import json
from DCA.vocabulary import Vocabulary

# # if gpu is to be used
# use_cuda = torch.cuda.is_available()
#
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
# Tensor = FloatTensor


def load(path, model_class, suffix=''):
    with io.open(path + '.config', 'r', encoding='utf8') as f:
        config = json.load(f)

    word_voca = Vocabulary()
    word_voca.__dict__ = config['word_voca']
    config['word_voca'] = word_voca

    entity_voca = Vocabulary()
    entity_voca.__dict__ = config['entity_voca']
    config['entity_voca'] = entity_voca

    model = model_class(config)
    model.load_state_dict(torch.load(path + '.state_dict' + suffix))

    return model


class AbstractWordEntity(nn.Module):
    """
    abstract class containing word and entity embeddings and vocabulary
    """

    def __init__(self, config=None):
        print('--- create AbstractWordEntity model ---')

        super(AbstractWordEntity, self).__init__()
        if config is None:
            return

        self.emb_dims = config['emb_dims']
        self.word_voca = config['word_voca']
        self.entity_voca = config['entity_voca']
        self.freeze_embs = config['freeze_embs']

        self.word_embeddings = config['word_embeddings_class'](self.word_voca.size(), self.emb_dims)
        self.entity_embeddings = config['entity_embeddings_class'](self.entity_voca.size(), self.emb_dims)

        if 'word_embeddings' in config:
            self.word_embeddings.weight = nn.Parameter(torch.Tensor(config['word_embeddings']))
        if 'entity_embeddings' in config:
            self.entity_embeddings.weight = nn.Parameter(torch.Tensor(config['entity_embeddings']))

        if self.freeze_embs:
            self.word_embeddings.weight.requires_grad = False
            self.entity_embeddings.weight.requires_grad = False

    def print_weight_norm(self):
        pass

    def save(self, path, suffix='', save_config=True):
        torch.save(self.state_dict(), path + '.state_dict' + suffix)

        if save_config:
            config = {'word_voca': self.word_voca.__dict__,
                      'entity_voca': self.entity_voca.__dict__}

            for k, v in self.__dict__.items():
                if not hasattr(v, '__dict__'):
                    config[k] = v

            with io.open(path + '.config', 'w', encoding='utf8') as f:
                json.dump(config, f)

    def load_params(self, path, param_names):
        params = torch.load(path)
        for pname in param_names:
            self._parameters[pname].data = params[pname]

    def loss(self, scores, grth):
        pass