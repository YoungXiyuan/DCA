import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from DCA.local_ctx_att_ranker import LocalCtxAttRanker
from torch.distributions import Categorical


class MulRelRanker(LocalCtxAttRanker):
    """
    multi-relational global model with context token attention, using loopy belief propagation
    """

    def __init__(self, config):

        print('--- create MulRelRanker model ---')
        super(MulRelRanker, self).__init__(config)
        self.dr = config['dr']
        self.gamma = config['gamma']
        self.tok_top_n4ent = config['tok_top_n4ent']
        self.tok_top_n4word = config['tok_top_n4word']
        self.tok_top_n4inlink = config['tok_top_n4inlink']
        self.dynamic = config['dynamic']

        self.ent_unk_id = config['entity_voca'].unk_id
        self.word_unk_id = config['word_voca'].unk_id
        self.ent_inlinks = config['entity_inlinks']

        # self.oracle = config.get('oracle', False)
        self.use_local = config.get('use_local', False)
        self.use_local_only = config.get('use_local_only', False)
        self.freeze_local = config.get('freeze_local', False)

        self.entity2entity_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))
        self.knowledge2entity_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))

        self.saved_log_probs = []
        self.rewards = []
        self.actions = []

        if self.freeze_local:
            self.att_mat_diag.requires_grad = False
            self.tok_score_mat_diag.requires_grad = False
        self.type_emb = torch.nn.Parameter(torch.randn([4, 5]))
        self.score_combine = torch.nn.Sequential(
                torch.nn.Linear(5, self.hid_dims),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.dr),
                torch.nn.Linear(self.hid_dims, 1))

        # print('---------------- model config -----------------')
        # for k, v in self.__dict__.items():
        #     if not hasattr(v, '__dict__'):
        #         print(k, v)
        # print('-----------------------------------------------')

    def forward(self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m, mtype, etype, gold=None, method="SL", isTrain=True):
        n_ments, n_cands = entity_ids.size()

        # if not self.oracle:
        #     gold = None
        self.mt_emb = torch.matmul(mtype, self.type_emb).view(n_ments, 1, -1)
        self.et_emb = torch.matmul(etype.view(-1, 4), self.type_emb).view(n_ments, n_cands, -1)
        tm = torch.sum(self.mt_emb*self.et_emb, -1, True)

        if self.use_local:
            local_ent_scores = super(MulRelRanker, self).forward(token_ids, tok_mask,
                                                                 entity_ids, entity_mask,
                                                                 p_e_m=None)
            ent_vecs = self._entity_vecs
        else:
            ent_vecs = self.entity_embeddings(entity_ids)
            local_ent_scores = Variable(torch.zeros(n_ments, n_cands).cuda(), requires_grad=False)

        # if self.use_local_only:
        #     inputs = torch.cat([local_ent_scores.view(n_ments * n_cands, -1),
        #                         torch.log(p_e_m + 1e-20).view(n_ments * n_cands, -1)], dim=1)
        #
        #     scores = self.score_combine(inputs).view(n_ments, n_cands)
        #
        #     return scores

        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.actions[:]

        scores = None
        cumulative_entity_ids = Variable(torch.LongTensor([self.ent_unk_id]).cuda())
        cumulative_knowledge_ids = Variable(torch.LongTensor([self.word_unk_id]).cuda())

        for i in range(n_ments):
            ent_coherence = self.compute_coherence(cumulative_entity_ids, entity_ids[i], entity_mask[i], self.entity2entity_mat_diag,
                                                   self.tok_top_n4ent, isWord=False)
            kng_coherence = self.compute_coherence(cumulative_knowledge_ids, entity_ids[i], entity_mask[i], self.knowledge2entity_mat_diag,
                                                   self.tok_top_n4word, isWord=True)

            inputs = torch.cat([local_ent_scores[i].view(n_cands, -1),
                                torch.log(p_e_m[i] + 1e-20).view(n_cands, -1), ent_coherence.view(n_cands, -1),
                                tm[i].view(n_cands, -1),
                                kng_coherence.view(n_cands, -1)*0], dim=1)

            score = self.score_combine(inputs).view(1, n_cands)
            action_prob = F.softmax(score - torch.max(score), 1)

            if isTrain:
                if method == "SL":
                    # print("gold[i]: ", gold[i])
                    # print("gold.data[i][0]: ", gold.data[i][0])

                    cumulative_entity_ids = torch.cat([cumulative_entity_ids, entity_ids[i][gold.data[i][0]]], dim=0)
                    cumulative_entity_ids = Variable(self.unique(cumulative_entity_ids.cpu().data.numpy()).cuda())

                    if (entity_ids[i][gold.data[i][0]]).data[0] in self.ent_inlinks:

                        external_inlinks = np.asarray(self.ent_inlinks[(entity_ids[i][gold.data[i][0]]).data[0]][:self.tok_top_n4inlink])

                        cumulative_knowledge_ids = Variable(self.unique(
                            np.concatenate((cumulative_knowledge_ids.cpu().data.numpy(), external_inlinks), axis=0)).cuda())

                    # debug
                    # print("Ground Truth:", gold.data[i][0])
                    #
                    # if (entity_ids[i][gold.data[i][0]]).data[0] in self.ent_inlinks:
                    #     print("Entity inLinks", self.ent_inlinks[(entity_ids[i][gold.data[i][0]]).data[0]])
                    #
                    # print("cumulative_entity_ids Size: ", cumulative_entity_ids)
                    # print("cumulative_knowledge_ids Size: ", cumulative_knowledge_ids)

                elif method == "RL":
                    m = Categorical(action_prob)
                    action = m.sample()

                    cumulative_entity_ids = torch.cat([cumulative_entity_ids, entity_ids[i][action.data[0]]], dim=0)
                    cumulative_entity_ids = Variable(self.unique(cumulative_entity_ids.cpu().data.numpy()).cuda())

                    if (entity_ids[i][action.data[0]]).data[0] in self.ent_inlinks:
                        # external_inlinks = Variable(torch.LongTensor(self.ent_inlinks[(entity_ids[i][action.data[0]]).data[0]][:self.tok_top_n4inlink]).cuda())
                        #
                        # cumulative_knowledge_ids = torch.cat([cumulative_knowledge_ids, external_inlinks], dim=0)
                        #
                        # cumulative_knowledge_ids = Variable(self.unique(cumulative_knowledge_ids.data).cuda)

                        external_inlinks = np.asarray(self.ent_inlinks[(entity_ids[i][action.data[0]]).data[0]][:self.tok_top_n4inlink])

                        cumulative_knowledge_ids = Variable(self.unique(
                            np.concatenate((cumulative_knowledge_ids.cpu().data.numpy(), external_inlinks), axis=0)).cuda())

                    # debug
                    # print("Ground Truth:", action.data[0])
                    #
                    # if (entity_ids[i][action.data[0]]).data[0] in self.ent_inlinks:
                    #     print("Entity inLinks", self.ent_inlinks[(entity_ids[i][action.data[0]]).data[0]])
                    #
                    # print("cumulative_entity_ids Size: ", cumulative_entity_ids)
                    # print("cumulative_knowledge_ids Size: ", cumulative_knowledge_ids)

                    self.saved_log_probs.append(m.log_prob(action))
                    # print("cumulative_knowledge_ids Size: ", cumulative_knowledge_ids)

                    self.saved_log_probs.append(m.log_prob(action))
                    self.actions.append(action.data[0])

                    if action.data[0] == gold.data[i][0]:
                        self.rewards.append(0)
                    else:
                        self.rewards.append(-1.)

            else:
                val, action = torch.max(action_prob, 1)

                # print("gold[action]: ", gold[action])
                # print("gold.data[action][0]: ", gold.data[action][0])

                cumulative_entity_ids = torch.cat([cumulative_entity_ids, entity_ids[i][action.data[0]]], dim=0)
                cumulative_entity_ids = Variable(self.unique(cumulative_entity_ids.cpu().data.numpy()).cuda())

                if (entity_ids[i][action.data[0]]).data[0] in self.ent_inlinks:

                    # external_inlinks = Variable(torch.LongTensor(self.ent_inlinks[(entity_ids[i][action.data[0]]).data[0]][:self.tok_top_n4inlink]).cuda())
                    #
                    # cumulative_knowledge_ids = torch.cat([cumulative_knowledge_ids, external_inlinks], dim=0)
                    #
                    # cumulative_knowledge_ids = Variable(self.unique(cumulative_knowledge_ids.data))
                    external_inlinks = np.asarray(self.ent_inlinks[(entity_ids[i][action.data[0]]).data[0]][:self.tok_top_n4inlink])

                    cumulative_knowledge_ids = Variable(self.unique(np.concatenate((cumulative_knowledge_ids.cpu().data.numpy(), external_inlinks), axis=0)).cuda())

                if method == "RL":
                    self.actions.append(action.data[0])

            if i == 0:
                scores = score
            else:
                scores = torch.cat([scores, score], dim=0)

        return scores, self.actions

    def unique(self, numpy_array):
        t = np.unique(numpy_array)
        return torch.from_numpy(t)

    def compute_coherence(self, cumulative_ids, entity_ids, entity_mask, att_mat_diag, window_size, isWord=False):
        n_cumulative_entities = cumulative_ids.size(0)
        n_entities = entity_ids.size(0)

        if isWord:
            cumulative_entity_vecs = self.word_embeddings(cumulative_ids)
        else:
            cumulative_entity_vecs = self.entity_embeddings(cumulative_ids)
        #cumulative_entity_vecs = self.entity_embeddings(cumulative_ids)

        entity_vecs = self.entity_embeddings(entity_ids)

        # debug
        # print("Cumulative_entity_ids Size: ", cumulative_ids.size(), cumulative_ids.size(0))
        # print("Entity_ids Size: ", entity_ids.size(), entity_ids.size(0))
        # print("Cumulative_entity_vecs Size: ", cumulative_entity_vecs.size())
        # print("Entity_vecs Size: ", entity_vecs.size())

        # att
        ent2ent_att_scores = torch.mm(entity_vecs * att_mat_diag, cumulative_entity_vecs.permute(1, 0))
        ent_tok_att_scores, _ = torch.max(ent2ent_att_scores, dim=0)
        top_tok_att_scores, top_tok_att_ids = torch.topk(ent_tok_att_scores, dim=0, k=min(window_size, n_cumulative_entities))

        # print("Top_tok_att_scores Size: ", top_tok_att_scores.size())
        # print("Top_tok_att_scores: ", top_tok_att_scores)

        entity_att_probs = F.softmax(top_tok_att_scores, dim=0).view(-1, 1)
        entity_att_probs = entity_att_probs / torch.sum(entity_att_probs, dim=0, keepdim=True)

        # print("entity_att_probs: ", entity_att_probs)

        selected_tok_vecs = torch.gather(cumulative_entity_vecs, dim=0,
                                         index=top_tok_att_ids.view(-1, 1).repeat(1, cumulative_entity_vecs.size(1)))

        ctx_ent_vecs = torch.sum((selected_tok_vecs * att_mat_diag) * entity_att_probs, dim=0, keepdim=True)

        # print("Selected_vecs * diag Size: ", (selected_tok_vecs * att_mat_diag).size())
        # print("Before Sum Size: ", ((selected_tok_vecs * att_mat_diag) * entity_att_probs).size())
        # print("Ctx_ent_vecs Size: ", ctx_ent_vecs.size())

        ent_ctx_scores = torch.mm(entity_vecs, ctx_ent_vecs.permute(1, 0)).view(-1, n_entities)

        # print("Ent_ctx_scores", ent_ctx_scores)

        scores = (ent_ctx_scores * entity_mask).add_((entity_mask - 1).mul_(1e10))

        # print("Scores: ", scores)

        return scores

    def print_weight_norm(self):
        LocalCtxAttRanker.print_weight_norm(self)

        print('f - l1.w, b', self.score_combine[0].weight.data.norm(), self.score_combine[0].bias.data.norm())
        print('f - l2.w, b', self.score_combine[3].weight.data.norm(), self.score_combine[3].bias.data.norm())

        # print(self.ctx_layer[0].weight.data.norm(), self.ctx_layer[0].bias.data.norm())
        # print('relations', self.rel_embs.data.norm(p=2, dim=1))
        # X = F.normalize(self.rel_embs)
        # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).sqrt()
        # print(diff)
        #
        # print('ew_embs', self.ew_embs.data.norm(p=2, dim=1))
        # X = F.normalize(self.ew_embs)
        # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).sqrt()
        # print(diff)

    def regularize(self, max_norm=4):
        # super(MulRelRanker, self).regularize(max_norm)
        # print("----MulRelRanker Regularization----")

        l1_w_norm = self.score_combine[0].weight.norm()
        l1_b_norm = self.score_combine[0].bias.norm()
        l2_w_norm = self.score_combine[3].weight.norm()
        l2_b_norm = self.score_combine[3].bias.norm()

        if (l1_w_norm > max_norm).data.all():
            self.score_combine[0].weight.data = self.score_combine[0].weight.data * max_norm / l1_w_norm.data
        if (l1_b_norm > max_norm).data.all():
            self.score_combine[0].bias.data = self.score_combine[0].bias.data * max_norm / l1_b_norm.data
        if (l2_w_norm > max_norm).data.all():
            self.score_combine[3].weight.data = self.score_combine[3].weight.data * max_norm / l2_w_norm.data
        if (l2_b_norm > max_norm).data.all():
            self.score_combine[3].bias.data = self.score_combine[3].bias.data * max_norm / l2_b_norm.data

    def finish_episode(self):
        policy_loss = []
        rewards = []

        # we only give a non-zero reward when done
        g_return = sum(self.rewards) / len(self.rewards)

        # add the final return in the last step
        rewards.insert(0, g_return)

        R = g_return
        for idx in range(len(self.rewards) - 1):
            R = R * self.gamma
            rewards.insert(0, R)

        rewards = torch.from_numpy(np.array(rewards)).type(torch.cuda.FloatTensor)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        policy_loss = torch.cat(policy_loss).sum()

        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.actions[:]

        return policy_loss

    def loss(self, scores, true_pos, method="SL", lamb=1e-7):
        loss = None

        # print("----MulRelRanker Loss----")
        if method == "SL":
            loss = F.multi_margin_loss(scores, true_pos, margin=self.margin)
        elif method == "RL":
            loss = self.finish_episode()

        # if self.use_local_only:
        #     return loss

        # regularization
        # X = F.normalize(self.rel_embs)
        # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).add_(1e-5).sqrt()
        # diff = diff * (diff < 1).float()
        # loss -= torch.sum(diff).mul(lamb)
        #
        # X = F.normalize(self.ew_embs)
        # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).add_(1e-5).sqrt()
        # diff = diff * (diff < 1).float()
        # loss -= torch.sum(diff).mul(lamb)
        return loss
