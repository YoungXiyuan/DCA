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
import json
import time

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
        self.one_entity_once = config['one_entity_once']
        self.seq_len = config['seq_len']

        self.word_vocab = config['word_voca']
        self.ent_vocab = config['entity_voca']

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
            config['use_local_only'] = self.args.use_local_only
            config['oracle'] = False
            self.model = ModelClass(config)

        self.load_ent_desc(500, 3)

        self.prerank_model.cuda()
        self.model.cuda()

    def load_ent_desc(self, max_desc_len, n_grams):
        ent_desc = json.load(open('../data/ent2desc.json', 'r'))
        self.ent_desc = [[self.word_vocab.get_id(Vocabulary.unk_token) for j in range(max_desc_len)] for i in range(self.ent_vocab.size())]
        self.desc_mask = [[0 for j in range(max_desc_len-n_grams+1)] for i in range(self.ent_vocab.size())]
        for ent in ent_desc:
            for i in range(min(len(ent_desc[ent]), max_desc_len)):
                self.ent_desc[self.ent_vocab.get_id(ent)][i] = self.word_vocab.get_id(ent_desc[ent][i])
                if (i>=n_grams-1):
                    self.desc_mask[self.ent_vocab.get_id(ent)][i-(n_grams-1)] = 1
        self.ent_desc = Variable(torch.LongTensor(self.ent_desc).cuda())
        self.desc_mask = Variable(torch.LongTensor(self.desc_mask).cuda())


    def get_data_items(self, dataset, predict=False, isTrain=False):
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

                if self.seq_len == 0:
                    if len(items) > 100:
                        print(len(items))
                        for k in range(0, len(items), 100):
                            data.append(items[k:min(len(items), k + 100)])
                    else:
                        data.append(items)
                else:
                # ----new implementation----
                # each doc is regarded as one batch
                # data.append(items)
                    if isTrain:
                        for k in range(0, len(items), self.seq_len // 2):
                            data.append(items[max(0, k - self.seq_len//2) : min(len(items), k + self.seq_len//2)])
                    else:
                        if self.one_entity_once:
                            for k in range(0, len(items)):
                                data.append(items[max(0, k-self.seq_len+1): k+1])
                        else:
                            for k in range(0, len(items), self.seq_len):
                                data.append(items[k:min(len(items), k + self.seq_len)])


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
                    if idx>len(m['cands'])-1:
                        continue
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

    # Automatic Order Learning Method - original version
    # def train(self, org_train_dataset, org_dev_datasets, config):
    #     print('extracting training data')
    #     train_dataset = self.get_data_items(org_train_dataset, predict=False)
    #     print('#train docs', len(train_dataset))
    #
    #     dev_datasets = []
    #     for dname, data in org_dev_datasets:
    #         dev_datasets.append((dname, self.get_data_items(data, predict=True)))
    #         print(dname, '#dev docs', len(dev_datasets[-1][1]))
    #
    #     print('creating optimizer')
    #     optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])
    #
    #     for param_name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             print(param_name)
    #
    #     best_f1 = -1
    #     not_better_count = 0
    #     is_counting = False
    #     order_learning = False
    #     eval_after_n_epochs = self.args.eval_after_n_epochs
    #
    #     rl_acc_threshold = 0.5
    #     order_learning_count = 0
    #
    #     # optimize the parameters within the disambiguation module first
    #     self.model.switch_order_learning(0)
    #
    #     for e in range(config['n_epochs']):
    #         if self.args.method == "SL" or self.args.method == "RL":
    #             shuffle(train_dataset)
    #
    #         total_loss = 0
    #
    #         if order_learning:
    #             order_learning_count += 1
    #
    #         if order_learning_count > 5:
    #             self.model.switch_order_learning(1)
    #
    #         for dc, batch in enumerate(train_dataset):  # each document is a minibatch
    #             self.model.train()
    #
    #             # convert data items to pytorch inputs
    #             token_ids = [m['context'][0] + m['context'][1]
    #                          if len(m['context'][0]) + len(m['context'][1]) > 0
    #                          else [self.model.word_voca.unk_id]
    #                          for m in batch]
    #
    #             ment_ids = [m['ment_ids'] if len(m['ment_ids']) > 0
    #                         else [self.model.word_voca.unk_id]
    #                         for m in batch]
    #
    #             entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).cuda())
    #             true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).cuda())
    #             p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).cuda())
    #             entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).cuda())
    #
    #             mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).cuda())
    #             etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).cuda())
    #
    #             token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)
    #             token_ids = Variable(torch.LongTensor(token_ids).cuda())
    #             token_mask = Variable(torch.FloatTensor(token_mask).cuda())
    #
    #             ment_ids, ment_mask = utils.make_equal_len(ment_ids, self.model.word_voca.unk_id)
    #             ment_ids = Variable(torch.LongTensor(ment_ids).cuda())
    #             ment_mask = Variable(torch.FloatTensor(ment_mask).cuda())
    #
    #             if self.args.method == "SL":
    #                 optimizer.zero_grad()
    #
    #                 # get the model output
    #                 if order_learning_count > 5 and len(batch) > 1:
    #                     scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype,
    #                                                    etype,
    #                                                    ment_ids, ment_mask, gold=true_pos.view(-1, 1),
    #                                                    method=self.args.method,
    #                                                    isTrain=False, isDynamic=0, isOrderLearning=order_learning,
    #                                                    isOrderFixed=False)
    #
    #                     loss = self.model.order_loss()
    #
    #                 else:
    #                     scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype,
    #                                                    ment_ids, ment_mask, gold=true_pos.view(-1, 1), method=self.args.method,
    #                                                    isTrain=True, isDynamic=0, isOrderLearning=order_learning,
    #                                                    isOrderFixed=True)
    #                     if order_learning:
    #                         _, targets = self.model.get_order_truth()
    #                         targets = Variable(torch.LongTensor(targets).cuda())
    #
    #                         if scores.size(0) != targets.size(0):
    #                             print("Size mismatch!")
    #                             break
    #                         loss = self.model.loss(scores, targets, method=self.args.method)
    #                     else:
    #                         loss = self.model.loss(scores, true_pos, method=self.args.method)
    #
    #                 loss.backward()
    #                 optimizer.step()
    #                 self.model.regularize(max_norm=4)
    #
    #                 loss = loss.cpu().data.numpy()
    #                 total_loss += loss
    #
    #             elif self.args.method == "RL":
    #                 action_memory = []
    #                 early_stop_count = 0
    #
    #                 for i_episode in count(1):  # the actual episode number for one doc is determined by decision accuracy
    #                     optimizer.zero_grad()
    #
    #                     # get the model output
    #                     if order_learning_count > 5 and len(batch) > 1:
    #                         scores, actions = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m,
    #                                                              mtype, etype,
    #                                                              ment_ids, ment_mask, gold=true_pos.view(-1, 1),
    #                                                              method=self.args.method,
    #                                                              isTrain=True, isDynamic=0,
    #                                                              isOrderLearning=order_learning,
    #                                                              isOrderFixed=False)
    #
    #                         loss = self.model.order_loss()
    #
    #                     else:
    #                         scores, actions = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m,
    #                                                              mtype, etype,
    #                                                              ment_ids, ment_mask, gold=true_pos.view(-1, 1),
    #                                                              method=self.args.method,
    #                                                              isTrain=True, isDynamic=0,
    #                                                              isOrderLearning=order_learning,
    #                                                              isOrderFixed=True)
    #                         if order_learning:
    #                             _, targets = self.model.get_order_truth()
    #                             targets = Variable(torch.LongTensor(targets).cuda())
    #
    #                             if scores.size(0) != targets.size(0):
    #                                 print("Size mismatch!")
    #                                 break
    #
    #                             loss = self.model.loss(scores, targets, method=self.args.method)
    #                         else:
    #                             loss = self.model.loss(scores, true_pos, method=self.args.method)
    #
    #                     loss.backward()
    #                     optimizer.step()
    #                     self.model.regularize(max_norm=4)
    #
    #                     loss = loss.cpu().data.numpy()
    #                     total_loss += loss
    #
    #                     # compute accuracy
    #                     correct = 0
    #                     total = 0.
    #                     if order_learning:
    #                         _, targets = self.model.get_order_truth()
    #                         for i in range(len(actions)):
    #                             if targets[i] == actions[i]:
    #                                 correct += 1
    #                             total += 1
    #                     else:
    #                         for i in range(len(actions)):
    #                             if true_pos.data[i] == actions[i]:
    #                                 correct += 1
    #                             total += 1
    #
    #                     if i_episode > len(batch):
    #                         break
    #
    #                     if actions == action_memory:
    #                         early_stop_count += 1
    #                     else:
    #                         del action_memory[:]
    #                         action_memory = copy.deepcopy(actions)
    #                         early_stop_count = 0
    #
    #                     if correct/total >= rl_acc_threshold or early_stop_count >= 5:
    #                         break
    #
    #             # print('epoch', e, "%0.2f%%" % (dc / len(train_dataset) * 100), loss)
    #
    #         if order_learning_count == 10:
    #             # Reset to the initial state
    #             order_learning_count = 0
    #             self.model.switch_order_learning(0)
    #
    #         print('epoch', e, 'total loss', total_loss, total_loss / len(train_dataset), flush=True)
    #
    #         if (e + 1) % eval_after_n_epochs == 0:
    #             dev_f1 = 0
    #
    #             if rl_acc_threshold < 0.95:
    #                 rl_acc_threshold += 0.05
    #
    #             for di, (dname, data) in enumerate(dev_datasets):
    #                 predictions = self.predict(data, 0, order_learning)
    #                 f1 = D.eval(org_dev_datasets[di][1], predictions)
    #
    #                 # predictions_1 = self.predict(data, 1, order_learning)
    #                 # f1_1 = D.eval(org_dev_datasets[di][1], predictions_1)
    #                 #
    #                 # predictions_2 = self.predict(data, 2, order_learning)
    #                 # f1_2 = D.eval(org_dev_datasets[di][1], predictions_2)
    #
    #                 print(dname, utils.tokgreen('micro F1: ' + str(f1)), flush=True)
    #
    #                 with open(self.output_path, 'a') as eval_csv_f1:
    #                     eval_f1_csv_writer = csv.writer(eval_csv_f1)
    #                     eval_f1_csv_writer.writerow([dname, e, 0, f1])
    #                     # eval_f1_csv_writer.writerow([dname, e, 1, f1_1])
    #                     # eval_f1_csv_writer.writerow([dname, e, 2, f1_2])
    #
    #                 if dname == 'aida-A':
    #                     dev_f1 = f1
    #
    #             if config['lr'] == 2e-4 and dev_f1 >= self.args.dev_f1_change_lr:
    #                 eval_after_n_epochs = 2
    #                 is_counting = True
    #                 best_f1 = dev_f1
    #                 not_better_count = 0
    #
    #                 self.model.switch_order_learning(0)
    #                 config['lr'] = 5e-5
    #                 print('change learning rate to', config['lr'])
    #                 optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])
    #
    #                 for param_name, param in self.model.named_parameters():
    #                     if param.requires_grad:
    #                         print(param_name)
    #
    #             if dev_f1 >= 0.92 and self.args.order_learning:
    #                 order_learning = True
    #
    #             if is_counting:
    #                 if dev_f1 < best_f1:
    #                     not_better_count += 1
    #                 else:
    #                     not_better_count = 0
    #                     best_f1 = dev_f1
    #                     print('save model to', self.args.model_path)
    #                     self.model.save(self.args.model_path)
    #
    #             if not_better_count == self.args.n_not_inc:
    #                 break
    #
    #             self.model.print_weight_norm()

    # Automatic Order Learning Method - final version
    # def train(self, org_train_dataset, org_dev_datasets, config):
    #     print('extracting training data')
    #     train_dataset = self.get_data_items(org_train_dataset, predict=False)
    #     print('#train docs', len(train_dataset))
    #
    #     dev_datasets = []
    #     for dname, data in org_dev_datasets:
    #         dev_datasets.append((dname, self.get_data_items(data, predict=True)))
    #         print(dname, '#dev docs', len(dev_datasets[-1][1]))
    #
    #     print('creating optimizer')
    #     optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])
    #
    #     for param_name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             print(param_name)
    #
    #     best_f1 = -1
    #     not_better_count = 0
    #     is_counting = False
    #     eval_after_n_epochs = self.args.eval_after_n_epochs
    #
    #     order_learning = False
    #     order_learning_count = 0
    #
    #     # optimize the parameters within the disambiguation module first
    #     self.model.switch_order_learning(0)
    #
    #     for e in range(config['n_epochs']):
    #         shuffle(train_dataset)
    #
    #         total_loss = 0
    #
    #         if order_learning:
    #             order_learning_count += 1
    #
    #         if order_learning_count > 5:
    #             self.model.switch_order_learning(1)
    #
    #         for dc, batch in enumerate(train_dataset):  # each document is a minibatch
    #             self.model.train()
    #
    #             # convert data items to pytorch inputs
    #             token_ids = [m['context'][0] + m['context'][1]
    #                          if len(m['context'][0]) + len(m['context'][1]) > 0
    #                          else [self.model.word_voca.unk_id]
    #                          for m in batch]
    #
    #             ment_ids = [m['ment_ids'] if len(m['ment_ids']) > 0
    #                         else [self.model.word_voca.unk_id]
    #                         for m in batch]
    #
    #             entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).cuda())
    #             true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).cuda())
    #             p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).cuda())
    #             entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).cuda())
    #
    #             mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).cuda())
    #             etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).cuda())
    #
    #             token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)
    #             token_ids = Variable(torch.LongTensor(token_ids).cuda())
    #             token_mask = Variable(torch.FloatTensor(token_mask).cuda())
    #
    #             ment_ids, ment_mask = utils.make_equal_len(ment_ids, self.model.word_voca.unk_id)
    #             ment_ids = Variable(torch.LongTensor(ment_ids).cuda())
    #             ment_mask = Variable(torch.FloatTensor(ment_mask).cuda())
    #
    #             optimizer.zero_grad()
    #
    #             # get the model output
    #             # alternate training for parameters within the order learning module and
    #             # parameters within the disambiguation module
    #             if order_learning_count > 5 and len(batch) > 1:
    #                 # when starting optimizing the parameters within the order learning module, we have to
    #                 # set isTrain=False to get the rewards from validation, for that SL always give the ground
    #                 # truth answer so that we can't identify whether the present parameters are good or not
    #                 scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype,
    #                                                etype, ment_ids, ment_mask, gold=true_pos.view(-1, 1),
    #                                                method=self.args.method,
    #                                                isTrain=False, isDynamic=0, isOrderLearning=order_learning,
    #                                                isOrderFixed=False)
    #
    #                 loss = self.model.order_loss()
    #
    #             else:
    #                 scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype,
    #                                                ment_ids, ment_mask, gold=true_pos.view(-1, 1),
    #                                                method=self.args.method,
    #                                                isTrain=True, isDynamic=0, isOrderLearning=order_learning,
    #                                                isOrderFixed=False)
    #                 if order_learning:
    #                     _, targets = self.model.get_order_truth()
    #                     targets = Variable(torch.LongTensor(targets).cuda())
    #
    #                     if scores.size(0) != targets.size(0):
    #                         print("Size mismatch!")
    #                         break
    #                     loss = self.model.loss(scores, targets, method=self.args.method)
    #                 else:
    #                     loss = self.model.loss(scores, true_pos, method=self.args.method)
    #
    #             loss.backward()
    #             optimizer.step()
    #             self.model.regularize(max_norm=4)
    #
    #             loss = loss.cpu().data.numpy()
    #             total_loss += loss
    #
    #             # print('epoch', e, "%0.2f%%" % (dc / len(train_dataset) * 100), loss)
    #
    #         if order_learning_count == 10:
    #             # Reset to the initial state
    #             order_learning_count = 0
    #             self.model.switch_order_learning(0)
    #
    #         print('epoch', e, 'total loss', total_loss, total_loss / len(train_dataset), flush=True)
    #
    #         if (e + 1) % eval_after_n_epochs == 0:
    #             dev_f1 = 0
    #
    #             for di, (dname, data) in enumerate(dev_datasets):
    #                 predictions = self.predict(data, 0, order_learning)
    #                 f1 = D.eval(org_dev_datasets[di][1], predictions)
    #
    #                 # predictions_1 = self.predict(data, 1, order_learning)
    #                 # f1_1 = D.eval(org_dev_datasets[di][1], predictions_1)
    #                 #
    #                 # predictions_2 = self.predict(data, 2, order_learning)
    #                 # f1_2 = D.eval(org_dev_datasets[di][1], predictions_2)
    #
    #                 print(dname, utils.tokgreen('micro F1: ' + str(f1)), flush=True)
    #
    #                 with open(self.output_path, 'a') as eval_csv_f1:
    #                     eval_f1_csv_writer = csv.writer(eval_csv_f1)
    #                     eval_f1_csv_writer.writerow([dname, e, 0, f1])
    #                     # eval_f1_csv_writer.writerow([dname, e, 1, f1_1])
    #                     # eval_f1_csv_writer.writerow([dname, e, 2, f1_2])
    #
    #                 if dname == 'aida-A':
    #                     dev_f1 = f1
    #
    #             if config['lr'] == 2e-4 and dev_f1 >= self.args.dev_f1_change_lr:
    #                 eval_after_n_epochs = 2
    #                 is_counting = True
    #                 best_f1 = dev_f1
    #                 not_better_count = 0
    #
    #                 self.model.switch_order_learning(0)
    #                 config['lr'] = 5e-5
    #                 print('change learning rate to', config['lr'])
    #                 optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])
    #
    #                 for param_name, param in self.model.named_parameters():
    #                     if param.requires_grad:
    #                         print(param_name)
    #
    #             if dev_f1 >= self.args.dev_f1_start_order_learning and self.args.order_learning:
    #                 order_learning = True
    #
    #             if is_counting:
    #                 if dev_f1 < best_f1:
    #                     not_better_count += 1
    #                 else:
    #                     not_better_count = 0
    #                     best_f1 = dev_f1
    #                     print('save model to', self.args.model_path)
    #                     self.model.save(self.args.model_path)
    #
    #             if not_better_count == self.args.n_not_inc:
    #                 break
    #
    #             self.model.print_weight_norm()

    # Heuristic Order Learning Method - Based on Mention-Local Similarity or Mention-Topical Similarity
    def train(self, org_train_dataset, org_dev_datasets, config):
        print('extracting training data')
        train_dataset = self.get_data_items(org_train_dataset, predict=False, isTrain=True)
        print('#train docs', len(train_dataset))
        self.init_lr = config['lr']
        dev_datasets = []
        for dname, data in org_dev_datasets:
            dev_datasets.append((dname, self.get_data_items(data, predict=True, isTrain=False)))
            print(dname, '#dev docs', len(dev_datasets[-1][1]))

        print('creating optimizer')
        optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])

        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                print(param_name)

        best_f1 = -1
        not_better_count = 0
        is_counting = False
        eval_after_n_epochs = self.args.eval_after_n_epochs

        order_learning = False
        # order_learning_count = 0

        rl_acc_threshold = 0.7

        # optimize the parameters within the disambiguation module first
        # self.model.switch_order_learning(0)
        best_aida_A_rlts = []
        best_aida_A_f1 = 0.
        best_aida_B_rlts = []
        best_aida_B_f1 = 0.
        best_ave_rlts = []
        best_ave_f1 = 0.
        # self.records = dict()
        # self.records[-1] = dict()
        # for di, (dname, data) in enumerate(dev_datasets):
        #     predictions = self.predict(data, config['isDynamic'], order_learning)
        #     self.records[-1][dname] = self.record
        # json.dump(self.records, open('records.json', 'w'),indent=4)
        self.run_time = []
        for e in range(config['n_epochs']):
            shuffle(train_dataset)

            total_loss = 0

            # if order_learning:
            #     order_learning_count += 1
            #
            # if order_learning_count > 5:
            #     self.model.switch_order_learning(1)

            for dc, batch in enumerate(train_dataset):  # each document is a minibatch
                self.model.train()

                # convert data items to pytorch inputs
                token_ids = [m['context'][0] + m['context'][1]
                             if len(m['context'][0]) + len(m['context'][1]) > 0
                             else [self.model.word_voca.unk_id]
                             for m in batch]

                ment_ids = [m['ment_ids'] if len(m['ment_ids']) > 0
                            else [self.model.word_voca.unk_id]
                            for m in batch]

                entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).cuda())
                true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).cuda())
                p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).cuda())
                entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).cuda())

                desc_ids = torch.index_select(self.ent_desc, 0, entity_ids.view(-1)).view(entity_ids.size(0), entity_ids.size(1), -1)
                desc_mask = torch.index_select(self.desc_mask, 0, entity_ids.view(-1)).view(entity_ids.size(0), entity_ids.size(1), -1)

                mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).cuda())
                etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).cuda())

                token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)
                token_ids = Variable(torch.LongTensor(token_ids).cuda())
                token_mask = Variable(torch.FloatTensor(token_mask).cuda())

                ment_ids, ment_mask = utils.make_equal_len(ment_ids, self.model.word_voca.unk_id)
                ment_ids = Variable(torch.LongTensor(ment_ids).cuda())
                ment_mask = Variable(torch.FloatTensor(ment_mask).cuda())

                # get the model output
                # alternate training for parameters within the order learning module and
                # parameters within the disambiguation module
                # if order_learning_count > 5 and len(batch) > 1:
                #     # when starting optimizing the parameters within the order learning module, we have to
                #     # set isTrain=False to get the rewards from validation, for that SL always give the ground
                #     # truth answer so that we can't identify whether the present parameters are good or not
                #     scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype,
                #                                    etype, ment_ids, ment_mask, gold=true_pos.view(-1, 1),
                #                                    method=self.args.method,
                #                                    isTrain=False, isDynamic=0, isOrderLearning=order_learning,
                #                                    isOrderFixed=False)
                #
                #     loss = self.model.order_loss()
                #
                # else:
                #     scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype,
                #                                    ment_ids, ment_mask, gold=true_pos.view(-1, 1),
                #                                    method=self.args.method,
                #                                    isTrain=True, isDynamic=0, isOrderLearning=order_learning,
                #                                    isOrderFixed=False)
                #     if order_learning:
                #         _, targets = self.model.get_order_truth()
                #         targets = Variable(torch.LongTensor(targets).cuda())
                #
                #         if scores.size(0) != targets.size(0):
                #             print("Size mismatch!")
                #             break
                #         loss = self.model.loss(scores, targets, method=self.args.method)
                #     else:
                #         loss = self.model.loss(scores, true_pos, method=self.args.method)

                # Uniform Training Process
                # optimizer.zero_grad()
                # scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype,
                #                                ment_ids, ment_mask, gold=true_pos.view(-1, 1),
                #                                method=self.args.method,
                #                                isTrain=True, isDynamic=0, isOrderLearning=order_learning,
                #                                isOrderFixed=True, isSort=self.args.sort)
                # if order_learning:
                #     _, targets = self.model.get_order_truth()
                #     targets = Variable(torch.LongTensor(targets).cuda())
                #
                #     if scores.size(0) != targets.size(0):
                #         print("Size mismatch!")
                #         break
                #     loss = self.model.loss(scores, targets, method=self.args.method)
                # else:
                #     loss = self.model.loss(scores, true_pos, method=self.args.method)
                #
                # loss.backward()
                # optimizer.step()
                # self.model.regularize(max_norm=4)
                #
                # loss = loss.cpu().data.numpy()
                # total_loss += loss

                # print('epoch', e, "%0.2f%%" % (dc / len(train_dataset) * 100), loss)

                if self.args.method == "SL":
                    optimizer.zero_grad()

                    scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype,
                                                   ment_ids, ment_mask, desc_ids, desc_mask, gold=true_pos.view(-1, 1),
                                                   method=self.args.method,
                                                   isTrain=True, isDynamic=config['isDynamic'], isOrderLearning=order_learning,
                                                   isOrderFixed=True, isSort=self.args.sort)

                    if order_learning:
                        _, targets = self.model.get_order_truth()
                        targets = Variable(torch.LongTensor(targets).cuda())

                        if scores.size(0) != targets.size(0):
                            print("Size mismatch!")
                            break
                        loss = self.model.loss(scores, targets, method=self.args.method)
                    else:
                        loss = self.model.loss(scores, true_pos, method=self.args.method)

                    loss.backward()
                    optimizer.step()
                    self.model.regularize(max_norm=4)

                    loss = loss.cpu().data.numpy()
                    total_loss += loss

                elif self.args.method == "RL":
                    action_memory = []
                    early_stop_count = 0
                    
                    # the actual episode number for one doc is determined by decision accuracy
                    for i_episode in count(1):
                        optimizer.zero_grad()

                        # get the model output
                        scores, actions = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m,
                                                             mtype, etype,
                                                             ment_ids, ment_mask, desc_ids, desc_mask, gold=true_pos.view(-1, 1),
                                                             method=self.args.method,
                                                             isTrain=True, isDynamic=config['isDynamic'],
                                                             isOrderLearning=order_learning,
                                                             isOrderFixed=True, isSort=self.args.sort)
                        if order_learning:
                            _, targets = self.model.get_order_truth()
                            targets = Variable(torch.LongTensor(targets).cuda())

                            if scores.size(0) != targets.size(0):
                                print("Size mismatch!")
                                break

                            loss = self.model.loss(scores, targets, method=self.args.method)
                        else:
                            loss = self.model.loss(scores, true_pos, method=self.args.method)

                        loss.backward()
                        optimizer.step()
                        #self.model.regularize(max_norm=4)

                        loss = loss.cpu().data.numpy()
                        total_loss += loss

                        # compute accuracy
                        correct = 0
                        total = 0.
                        if order_learning:
                            _, targets = self.model.get_order_truth()
                            for i in range(len(actions)):
                                if targets[i] == actions[i]:
                                    correct += 1
                                total += 1
                        else:
                            for i in range(len(actions)):
                                if true_pos.data[i] == actions[i]:
                                    correct += 1
                                total += 1

                        if not config['use_early_stop']:
                            break

                        if i_episode > len(batch)/2:
                            break

                        if actions == action_memory:
                            early_stop_count += 1
                        else:
                            del action_memory[:]
                            action_memory = copy.deepcopy(actions)
                            early_stop_count = 0

                        if correct/total >= rl_acc_threshold or early_stop_count >= 3:
                            break

            # if order_learning_count == 10:
            #     # Reset to the initial state
            #     order_learning_count = 0
            #     self.model.switch_order_learning(0)

            print('epoch', e, 'total loss', total_loss, total_loss / len(train_dataset), flush=True)

            if (e + 1) % eval_after_n_epochs == 0:
                dev_f1 = 0.
                test_f1 = 0.
                ave_f1 = 0.
                if rl_acc_threshold < 0.92:
                    rl_acc_threshold += 0.02
                temp_rlt = []
                #self.records[e] = dict()
                for di, (dname, data) in enumerate(dev_datasets):
                    if dname == 'aida-B':
                        self.rt_flag = True
                    else:
                        self.rt_flag = False
                    predictions = self.predict(data, config['isDynamic'], order_learning)
                    #self.records[e][dname] = self.record
                    f1 = D.eval(org_dev_datasets[di][1], predictions)

                    # predictions_1 = self.predict(data, 1, order_learning)
                    # f1_1 = D.eval(org_dev_datasets[di][1], predictions_1)
                    #
                    # predictions_2 = self.predict(data, 2, order_learning)
                    # f1_2 = D.eval(org_dev_datasets[di][1], predictions_2)

                    print(dname, utils.tokgreen('micro F1: ' + str(f1)), flush=True)

                    with open(self.output_path, 'a') as eval_csv_f1:
                        eval_f1_csv_writer = csv.writer(eval_csv_f1)
                        eval_f1_csv_writer.writerow([dname, e, 0, f1])
                        #eval_f1_csv_writer.writerow([dname, e, 1, f1_1])
                        #eval_f1_csv_writer.writerow([dname, e, 2, f1_2])
                    temp_rlt.append([dname, f1])
                    if dname == 'aida-A':
                        dev_f1 = f1
                    if dname == 'aida-B':
                        test_f1 = f1
                    ave_f1 += f1
                if dev_f1>best_aida_A_f1:
                    best_aida_A_f1 = dev_f1
                    best_aida_A_rlts = copy.deepcopy(temp_rlt)
                if test_f1>best_aida_B_f1:
                    best_aida_B_f1 = test_f1
                    best_aida_B_rlts = copy.deepcopy(temp_rlt)
                if ave_f1 > best_ave_f1:
                    best_ave_f1 = ave_f1
                    best_ave_rlts = copy.deepcopy(temp_rlt)

                if not config['isDynamic']:
                    self.record_runtime('DCA')
                else:
                    self.record_runtime('local')

                #json.dump(self.records, open('records.json', 'w'), indent=4)
                if config['lr'] == self.init_lr and dev_f1 >= self.args.dev_f1_change_lr:
                    eval_after_n_epochs = 2
                    is_counting = True
                    best_f1 = dev_f1
                    not_better_count = 0

                    # self.model.switch_order_learning(0)
                    config['lr'] = self.init_lr / 2
                    print('change learning rate to', config['lr'])
                    optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])

                    for param_name, param in self.model.named_parameters():
                        if param.requires_grad:
                            print(param_name)

                if dev_f1 >= self.args.dev_f1_start_order_learning and self.args.order_learning:
                    order_learning = True

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

        print('best_aida_A_rlts', best_aida_A_rlts)
        print('best_aida_B_rlts', best_aida_B_rlts)
        print('best_ave_rlts', best_ave_rlts)

    def record_runtime(self, method):
        self.run_time.sort(key=lambda x:x[0])
        pre_cands = 0
        count = 0
        total = 0.
        rt = dict()
        for cands, ti in self.run_time:
            if not cands==pre_cands:
                if pre_cands > 0:
                    rt[pre_cands] = total / count
                total = ti
                count = 1
                pre_cands = cands
            else:
                count += 1
                total += ti
        if count >0:
            rt[pre_cands] = total / count
        with open('runtime_%s.csv'%method, 'w') as runtime_csv:
            runtime_csv_writer = csv.writer(runtime_csv)
            for cands, ti in rt.items():
                runtime_csv_writer.writerow([cands, ti])
            runtime_csv.close()

    def predict(self, data, dynamic_option, order_learning):
        predictions = {items[0]['doc_name']: [] for items in data}
        self.model.eval()
        #self.record = []
        for batch in data:  # each document is a minibatch
            start_time = time.time()
            token_ids = [m['context'][0] + m['context'][1]
                         if len(m['context'][0]) + len(m['context'][1]) > 0
                         else [self.model.word_voca.unk_id]
                         for m in batch]

            ment_ids = [m['ment_ids'] if len(m['ment_ids']) > 0
                        else [self.model.word_voca.unk_id]
                        for m in batch]

            total_candidates = sum([len(m['selected_cands']['cands']) for m in batch])

            entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).cuda())
            p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).cuda())
            entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).cuda())
            true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).cuda())

            token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)

            token_ids = Variable(torch.LongTensor(token_ids).cuda())
            token_mask = Variable(torch.FloatTensor(token_mask).cuda())

            desc_ids = torch.index_select(self.ent_desc, 0, entity_ids.view(-1)).view(entity_ids.size(0),
                                                                                      entity_ids.size(1), -1)
            desc_mask = torch.index_select(self.desc_mask, 0, entity_ids.view(-1)).view(entity_ids.size(0),
                                                                                        entity_ids.size(1), -1)
            ment_ids, ment_mask = utils.make_equal_len(ment_ids, self.model.word_voca.unk_id)
            ment_ids = Variable(torch.LongTensor(ment_ids).cuda())
            ment_mask = Variable(torch.FloatTensor(ment_mask).cuda())

            mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).cuda())
            etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).cuda())

            scores, actions = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype,
                                                 ment_ids, ment_mask, desc_ids, desc_mask, gold=true_pos.view(-1, 1),
                                                 method=self.args.method,
                                                 isTrain=False, isDynamic=dynamic_option,
                                                 isOrderLearning=order_learning,
                                                 isOrderFixed=True, isSort=self.args.sort)

            scores = scores.cpu().data.numpy()

            pred_ids = np.argmax(scores, axis=1)
            end_time = time.time()
            if self.rt_flag:
                self.run_time.append([total_candidates, end_time-start_time])
            if order_learning:
                pred_entities = list()

                decision_order, _ = self.model.get_order_truth()

                for mi, m in enumerate(batch):
                    pi = pred_ids[decision_order.index(mi)]
                    if m['selected_cands']['mask'][pi] == 1:
                        pred_entities.append(m['selected_cands']['named_cands'][pi])
                    else:
                        if m['selected_cands']['mask'][0] == 1:
                            pred_entities.append(m['selected_cands']['named_cands'][0])
                        else:
                            pred_entities.append('NIL')
            else:
                pred_entities = [m['selected_cands']['named_cands'][i] if m['selected_cands']['mask'][i] == 1
                                 else (m['selected_cands']['named_cands'][0] if m['selected_cands']['mask'][0] == 1 else 'NIL')
                                 for (i, m) in zip(pred_ids, batch)]

            doc_names = [m['doc_name'] for m in batch]
            self.added_words = []
            self.added_ents = []
            if self.seq_len>0 and self.one_entity_once:
                #self.added_words.append([self.word_vocab.id2word[idx] for idx in self.model.added_words[-1]])
                #self.added_ents.append([self.ent_vocab.id2word[idx] for idx in self.model.added_ents[-1]])
                predictions[doc_names[-1]].append({'pred': (pred_entities[-1], 0.)})
            else:
                # for ids in self.model.added_words:
                #     self.added_words.append([self.word_vocab.id2word[idx] for idx in ids])
                # for ids in self.model.added_ents:
                #     self.added_ents.append([self.ent_vocab.id2word[idx] for idx in ids])
                for dname, entity in zip(doc_names, pred_entities):
                    predictions[dname].append({'pred': (entity, 0.)})
            #self.record.append(dict({'added_words':self.added_words, 'added_ents':self.added_ents}))
        return predictions

