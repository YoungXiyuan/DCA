import numpy as np
from DCA.vocabulary import Vocabulary
import torch
from torch.autograd import Variable
import DCA.dataset as D
import DCA.utils as utils
import DCA.ntee as ntee
from random import shuffle
import torch.optim as optim
from DCA.abstract_word_entity import load as load_model
from DCA.mulrel_ranker import MulRelRanker
from pprint import pprint
from itertools import count
import copy
import csv


ModelClass = MulRelRanker
wiki_prefix = 'en.wikipedia.org/wiki/'


class EDRanker:
    """
    ranking candidates
    """
    def __init__(self, config):
        print('--- create EDRanker model ---')

        config['entity_embeddings'] = config['entity_embeddings'] / \
                                      np.maximum(np.linalg.norm(config['entity_embeddings'],
                                                                axis=1, keepdims=True), 1e-12)
        config['entity_embeddings'][config['entity_voca'].unk_id] = 1e-10
        config['word_embeddings'] = config['word_embeddings'] / \
                                    np.maximum(np.linalg.norm(config['word_embeddings'],
                                                              axis=1, keepdims=True), 1e-12)
        config['word_embeddings'][config['word_voca'].unk_id] = 1e-10

        self.output_path = config['f1_csv_path']

        print('prerank model')
        self.prerank_model = ntee.NTEE(config)
        self.args = config['args']

        print('main model')
        if self.args.mode == 'eval':
            print('try loading model from', self.args.model_path)
            self.model = load_model(self.args.model_path, ModelClass)
        else:
            print('create new model')

            config['use_local'] = True
            config['use_local_only'] = False
            config['oracle'] = False
            self.model = ModelClass(config)

        self.prerank_model.cuda()
        self.model.cuda()

    def get_data_items(self, dataset, predict=False):
        data = []
        cand_source = 'candidates'

        for doc_name, content in dataset.items():
            items = []

            for m in content:
                try:
                    named_cands = [c[0] for c in m[cand_source]]
                    p_e_m = [min(1., max(1e-3, c[1])) for c in m[cand_source]]
                    etype = [c[2] for c in m[cand_source]]
                except:
                    named_cands = [c[0] for c in m['candidates']]
                    p_e_m = [min(1., max(1e-3, c[1])) for c in m['candidates']]
                    etype = [c[2] for c in m['candidates']]
                try:
                    true_pos = named_cands.index(m['gold'][0])
                    p = p_e_m[true_pos]
                except:
                    true_pos = -1

                # select # n_cands_before_rank candidates according to their priority
                named_cands = named_cands[:min(self.args.n_cands_before_rank, len(named_cands))]
                p_e_m = p_e_m[:min(self.args.n_cands_before_rank, len(p_e_m))]
                etype = etype[:min(self.args.n_cands_before_rank, len(etype))]
                # guarantee that the ground truth is in the top30 candidates
                if true_pos >= len(named_cands):
                    if not predict:
                        true_pos = len(named_cands) - 1
                        p_e_m[-1] = p
                        named_cands[-1] = m['gold'][0]
                    else:
                        true_pos = -1

                cands = [self.model.entity_voca.get_id(wiki_prefix + c) for c in named_cands]
                mask = [1.] * len(cands)

                if len(cands) == 0 and not predict:
                    continue
                elif len(cands) < self.args.n_cands_before_rank:
                    cands += [self.model.entity_voca.unk_id] * (self.args.n_cands_before_rank - len(cands))
                    etype += [[0, 0, 0, 1]] * (self.args.n_cands_before_rank - len(etype))
                    named_cands += [Vocabulary.unk_token] * (self.args.n_cands_before_rank - len(named_cands))
                    p_e_m += [1e-8] * (self.args.n_cands_before_rank - len(p_e_m))
                    mask += [0.] * (self.args.n_cands_before_rank - len(mask))

                lctx = m['context'][0].strip().split()
                lctx_ids = [self.prerank_model.word_voca.get_id(t) for t in lctx if utils.is_important_word(t)]
                lctx_ids = [tid for tid in lctx_ids if tid != self.prerank_model.word_voca.unk_id]
                lctx_ids = lctx_ids[max(0, len(lctx_ids) - self.args.ctx_window//2):]

                rctx = m['context'][1].strip().split()
                rctx_ids = [self.prerank_model.word_voca.get_id(t) for t in rctx if utils.is_important_word(t)]
                rctx_ids = [tid for tid in rctx_ids if tid != self.prerank_model.word_voca.unk_id]
                rctx_ids = rctx_ids[:min(len(rctx_ids), self.args.ctx_window//2)]

                ment = m['mention'].strip().split()
                ment_ids = [self.prerank_model.word_voca.get_id(t) for t in ment if utils.is_important_word(t)]
                ment_ids = [tid for tid in ment_ids if tid != self.prerank_model.word_voca.unk_id]

                m['sent'] = ' '.join(lctx + rctx)
                mtype = m['mtype']
                items.append({'context': (lctx_ids, rctx_ids),
                              'ment_ids': ment_ids,
                              'cands': cands,
                              'named_cands': named_cands,
                              'p_e_m': p_e_m,
                              'mask': mask,
                              'true_pos': true_pos,
                              'mtype': mtype,
                              'etype': etype,
                              'doc_name': doc_name,
                              'raw': m
                              })

            if len(items) > 0:
                # note: this shouldn't affect the order of prediction because we use doc_name to add predicted entities,
                # and we don't shuffle the data for prediction

                # ----old implementation-----
                if len(items) > 100:
                    print(len(items))
                    for k in range(0, len(items), 100):
                        data.append(items[k:min(len(items), k + 100)])
                else:
                    data.append(items)

                # ----new implementation----
                # each doc is regarded as one batch
                # data.append(items)

        return self.prerank(data, predict)

    def prerank(self, dataset, predict=False):
        new_dataset = []
        has_gold = 0
        total = 0

        for content in dataset:
            items = []

            if self.args.keep_ctx_ent > 0:
                # rank the candidates by ntee scores
                lctx_ids = [m['context'][0][max(len(m['context'][0]) - self.args.prerank_ctx_window // 2, 0):]
                            for m in content]
                rctx_ids = [m['context'][1][:min(len(m['context'][1]), self.args.prerank_ctx_window // 2)]
                            for m in content]
                ment_ids = [[] for m in content]

                token_ids = [l + m + r if len(l) + len(r) > 0 else [self.prerank_model.word_voca.unk_id]
                             for l, m, r in zip(lctx_ids, ment_ids, rctx_ids)]

                entity_ids = [m['cands'] for m in content]
                entity_ids = Variable(torch.LongTensor(entity_ids).cuda())

                entity_mask = [m['mask'] for m in content]
                entity_mask = Variable(torch.FloatTensor(entity_mask).cuda())

                token_ids, token_offsets = utils.flatten_list_of_lists(token_ids)
                token_ids = Variable(torch.LongTensor(token_ids).cuda())
                token_offsets = Variable(torch.LongTensor(token_offsets).cuda())

                log_probs = self.prerank_model.forward(token_ids, token_offsets, entity_ids, use_sum=True)
                log_probs = (log_probs * entity_mask).add_((entity_mask - 1).mul_(1e10))
                _, top_pos = torch.topk(log_probs, dim=1, k=self.args.keep_ctx_ent)
                top_pos = top_pos.data.cpu().numpy()
            else:
                top_pos = [[]] * len(content)

            # select candidats: mix between keep_ctx_ent best candidates (ntee scores) with
            # keep_p_e_m best candidates (p_e_m scores)
            for i, m in enumerate(content):
                sm = {'cands': [],
                      'named_cands': [],
                      'p_e_m': [],
                      'mask': [],
                      'etype': [],
                      'true_pos': -1}
                m['selected_cands'] = sm

                selected = set(top_pos[i])
                idx = 0
                while len(selected) < self.args.keep_ctx_ent + self.args.keep_p_e_m:
                    if idx not in selected:
                        selected.add(idx)
                    idx += 1

                selected = sorted(list(selected))
                for idx in selected:
                    sm['cands'].append(m['cands'][idx])
                    sm['named_cands'].append(m['named_cands'][idx])
                    sm['p_e_m'].append(m['p_e_m'][idx])
                    sm['mask'].append(m['mask'][idx])
                    sm['etype'].append(m['etype'][idx])
                    if idx == m['true_pos']:
                        sm['true_pos'] = len(sm['cands']) - 1

                if not predict:
                    if sm['true_pos'] == -1:
                        continue

                        # this insertion only makes the performance worse (why???)
                        # sm['true_pos'] = 0
                        # sm['cands'][0] = m['cands'][m['true_pos']]
                        # sm['named_cands'][0] = m['named_cands'][m['true_pos']]
                        # sm['p_e_m'][0] = m['p_e_m'][m['true_pos']]
                        # sm['mask'][0] = m['mask'][m['true_pos']]
                        # if m['true_pos'] != -1:
                        #     sm['true_pos'] = len(selected) - 1
                        #     sm['cands'][-1] = m['cands'][m['true_pos']]
                        #     sm['named_cands'][-1] = m['named_cands'][m['true_pos']]
                        #     sm['p_e_m'][-1] = m['p_e_m'][m['true_pos']]
                        #     sm['mask'][-1] = m['mask'][m['true_pos']]

                items.append(m)
                if sm['true_pos'] >= 0:
                    has_gold += 1
                total += 1

                if predict:
                    # only for oracle model, not used for eval
                    if sm['true_pos'] == -1:
                        sm['true_pos'] = 0  # a fake gold, happens only 2%, but avoid the non-gold

            if len(items) > 0:
                new_dataset.append(items)

        print('recall', has_gold / total)
        return new_dataset

    def train(self, org_train_dataset, org_dev_datasets, config):
        print('extracting training data')
        train_dataset = self.get_data_items(org_train_dataset, predict=False)
        print('#train docs', len(train_dataset))

        dev_datasets = []
        for dname, data in org_dev_datasets:
            dev_datasets.append((dname, self.get_data_items(data, predict=True)))
            print(dname, '#dev docs', len(dev_datasets[-1][1]))

        print('creating optimizer')
        optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])
        best_f1 = -1
        not_better_count = 0
        is_counting = False
        eval_after_n_epochs = self.args.eval_after_n_epochs

        for e in range(config['n_epochs']):
            if self.args.method == "SL" or self.args.method == "RL":
                shuffle(train_dataset)

            total_loss = 0

            for dc, batch in enumerate(train_dataset):  # each document is a minibatch
                self.model.train()


                # convert data items to pytorch inputs
                token_ids = [m['context'][0] + m['context'][1]
                             if len(m['context'][0]) + len(m['context'][1]) > 0
                             else [self.model.word_voca.unk_id]
                             for m in batch]

                entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).cuda())
                true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).cuda())
                p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).cuda())
                entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).cuda())

                mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).cuda())
                etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).cuda())

                token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)
                token_ids = Variable(torch.LongTensor(token_ids).cuda())
                token_mask = Variable(torch.FloatTensor(token_mask).cuda())

                if self.args.method == "SL":
                    optimizer.zero_grad()

                    # get the model output
                    scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype,
                                                gold=true_pos.view(-1, 1), method=self.args.method, isTrain=True)

                    loss = self.model.loss(scores, true_pos, method=self.args.method)

                    loss.backward()
                    optimizer.step()
                    self.model.regularize(max_norm=4)

                    loss = loss.cpu().data.numpy()
                    total_loss += loss

                elif self.args.method == "RL":
                    action_memory = []
                    early_stop_count = 0

                    for i_episode in count(1):  # the actual episode number for one doc is determined by decision accuracy
                        optimizer.zero_grad()

                        # get the model output
                        scores, actions = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype,
                                                       gold=true_pos.view(-1, 1), method=self.args.method, isTrain=True)

                        # compute accuracy
                        correct = 0
                        total = 0.
                        for i in range(len(actions)):
                            if true_pos.data[i] == actions[i]:
                                correct += 1
                            total += 1

                        loss = self.model.loss(scores, true_pos, method=self.args.method)

                        loss.backward()
                        optimizer.step()

                        loss = loss.cpu().data.numpy()
                        total_loss += loss

                        if i_episode > len(batch):
                            break

                        if actions == action_memory:
                            early_stop_count += 1
                        else:
                            del action_memory[:]
                            action_memory = copy.deepcopy(actions)
                            early_stop_count = 0

                        if correct/total >= 0.8 or early_stop_count >= 5:
                            break

                # print('epoch', e, "%0.2f%%" % (dc / len(train_dataset) * 100), loss)

            print('epoch', e, 'total loss', total_loss, total_loss / len(train_dataset), flush=True)

            if (e + 1) % eval_after_n_epochs == 0:
                dev_f1 = 0
                for di, (dname, data) in enumerate(dev_datasets):
                    predictions = self.predict(data)
                    f1 = D.eval(org_dev_datasets[di][1], predictions)
                    print(dname, utils.tokgreen('micro F1: ' + str(f1)), flush=True)

                    with open(self.output_path, 'a') as eval_csv_f1:
                        eval_f1_csv_writer = csv.writer(eval_csv_f1)
                        eval_f1_csv_writer.writerow([dname, e, f1])

                    if dname == 'aida-A':
                        dev_f1 = f1

                if config['lr'] == 1e-4 and dev_f1 >= self.args.dev_f1_change_lr:
                    eval_after_n_epochs = 2
                    is_counting = True
                    best_f1 = dev_f1
                    not_better_count = 0

                    config['lr'] = 1e-5
                    print('change learning rate to', config['lr'])
                    optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])

                if is_counting:
                    if dev_f1 < best_f1:
                        not_better_count += 1
                    else:
                        not_better_count = 0
                        best_f1 = dev_f1
                        print('save model to', self.args.model_path)
                        self.model.save(self.args.model_path)

                if not_better_count == self.args.n_not_inc:
                    break

                self.model.print_weight_norm()

    def predict(self, data):
        predictions = {items[0]['doc_name']: [] for items in data}
        self.model.eval()

        for batch in data:  # each document is a minibatch
            token_ids = [m['context'][0] + m['context'][1]
                         if len(m['context'][0]) + len(m['context'][1]) > 0
                         else [self.model.word_voca.unk_id]
                         for m in batch]

            entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).cuda())
            p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).cuda())
            entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).cuda())
            true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).cuda())

            token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)

            token_ids = Variable(torch.LongTensor(token_ids).cuda())
            token_mask = Variable(torch.FloatTensor(token_mask).cuda())

            mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).cuda())
            etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).cuda())

            scores, actions = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype,
                                        gold=true_pos.view(-1, 1), method=self.args.method, isTrain=False)

            scores = scores.cpu().data.numpy()

            pred_ids = np.argmax(scores, axis=1)
            pred_entities = [m['selected_cands']['named_cands'][i] if m['selected_cands']['mask'][i] == 1
                             else (m['selected_cands']['named_cands'][0] if m['selected_cands']['mask'][0] == 1 else 'NIL')
                             for (i, m) in zip(pred_ids, batch)]

            doc_names = [m['doc_name'] for m in batch]

            for dname, entity in zip(doc_names, pred_entities):
                predictions[dname].append({'pred': (entity, 0.)})

        return predictions
