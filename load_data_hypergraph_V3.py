import numpy as np
import random as rd
import scipy.sparse as sp
import time
import pickle
import os
import pdb
from collections import defaultdict
# test

"""
额外添加 (I-A, U)超图;
"""
class Data(object):
    def __init__(self, path, batch_size):
        if 'taobao' in path:
            print('Data loader won\'t provide title feat.')
        self.path = path

        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        title_file = path + '/i2e_title.txt'
        review_file = path + '/i2e_review.txt'
        visual_file = path + '/i2e_visual.txt'
        all_file = path + '/i2e_all.txt'
        sentiment_file = path + '/item2sentiment.pickle'

        self.exist_items_in_entity = set()
        self.exist_items_in_title = set()
        self.exist_items_in_review = set()
        self.exist_items_in_visual = set()

        # get number of users and items
        self.n_users, self.n_items, self.n_entity = 0, 0, 0

        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        self.exist_title_entity, self.exist_review_entity, self.exist_visual_entity = set(), set(), set()

        # exist (user, item) pair
        self.item2user_l = defaultdict(list)
        self.exit_user_item_pair_l = []
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
                    self.exit_user_item_pair_l.extend([(uid, item) for item in items])
                    # dict
                    for item in items:
                        self.item2user_l[item].append(uid)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)

        self.item2title_entity = {}
        with open(title_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) != 1 and l[1:] != ['']:
                        i, e_list = l[0],l[1:]
                        e_list = [int(e) for e in e_list]
                        self.exist_title_entity |= set(e_list)
                        self.item2title_entity[int(i)] = e_list
                    else:
                        i = l[0]
                        self.item2title_entity[int(i)] = []

        self.item2review_entity = {}
        if os.path.exists(review_file):
            with open(review_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        if len(l) != 1 and l[1:] != ['']:
                            i, e_list = l[0], l[1:]
                            e_list = [int(e) for e in e_list]
                            self.exist_review_entity |= set(e_list)
                            self.item2review_entity[int(i)] = e_list
                        else:
                            i = l[0]
                            self.item2review_entity[int(i)] = []
        else:
            for i in range(self.n_items):
                self.item2review_entity[i] = []

        self.item2visual_entity = {}
        with open(visual_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) != 1 and l[1:] != ['']:
                        i, e_list = l[0], l[1:]
                        e_list = [int(e) for e in e_list]
                        self.exist_visual_entity |= set(e_list)
                        self.item2visual_entity[int(i)] = e_list
                    else:
                        i = l[0]
                        self.item2visual_entity[int(i)] = []

        self.exist_entity = set()
        self.item2entity = {}
        with open(all_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) != 1 and l[1:] != ['']:
                        i, e_list = l[0], l[1:]
                        e_list = [int(e) for e in e_list]
                        self.exist_entity |= set(e_list)
                        self.item2entity[int(i)] = e_list
                    else:
                        i = l[0]
                        self.item2entity[int(i)] = []

        self.n_items += 1
        self.n_users += 1
        self.n_entity = len(self.exist_entity)

        self.exist_items = list(range(self.n_items))

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.R_senti = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.title_R = sp.dok_matrix((self.n_items, self.n_entity), dtype=np.float32)
        self.review_R = sp.dok_matrix((self.n_items, self.n_entity), dtype=np.float32)
        self.visual_R = sp.dok_matrix((self.n_items, self.n_entity), dtype=np.float32)
        self.all_R = sp.dok_matrix((self.n_items, self.n_entity), dtype=np.float32)

        self.all_R_senti = sp.dok_matrix((self.n_items, self.n_entity), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        self.train_users = {}
        self.train_users_f = {}

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    for i in items:
                        if i not in self.train_users_f:
                            self.train_users_f[i] = []
                        else:
                            self.train_users_f[i].append(uid)

        max_inter_i = 0
        for i, us in self.train_users_f.items():
            max_inter_i = max(max_inter_i, len(us))
        if os.path.exists(sentiment_file):

            with open(sentiment_file, 'rb') as file:
                item2sentiment = pickle.load(file)

            self.item2sentiment_ = {}
            sum = 0.0

            for i_, s_list in item2sentiment.items():
                temp_list = []
                len_1 = len(s_list)
                for s in s_list:
                    if s == 'positive':
                        temp_list.append(1.0)
                    else:
                        temp_list.append(0.0)
                len_2 = len(temp_list)
                if len_1 != len_2:
                    print('something wrong!!')
                self.item2sentiment_[i_] = np.mean(temp_list) ** 0.1
                sum += self.item2sentiment_[i_]

            for i, s in self.item2sentiment_.items():
                self.item2sentiment_[i] = self.item2sentiment_[i] / sum * len(self.item2sentiment_)
        else:
            sum = 0.0
            self.item2sentiment_ = {}

            for i in range(self.n_items):

                if i not in self.train_users_f:
                    self.item2sentiment_[i] = 1

                else:
                    if len(self.train_users_f[i]) == 0:
                        self.item2sentiment_[i] = 1.0
                    else:
                        self.item2sentiment_[i] = (len(self.train_users_f[i])/max_inter_i) ** 0.01
                sum += self.item2sentiment_[i]

            for i, s in self.item2sentiment_.items():
                self.item2sentiment_[i] = self.item2sentiment_[i] / sum * self.n_items


        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        self.R_senti[uid, i] = self.item2sentiment_[i]

                        if i not in self.train_users:
                            self.train_users[i] = []
                        self.train_users[i].append(uid)

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        for i, l in self.item2title_entity.items():
            if len(l) > 1:
                self.exist_items_in_title.add(i)
                for e in l:
                    self.title_R[i, e] = 1.
                    self.all_R[i, e] = 1.

                    self.all_R_senti[i, e] = self.item2sentiment_[i]


        for i, l in self.item2review_entity.items():
            if len(l) > 1:
                self.exist_items_in_review.add(i)
                for e in l:
                    self.review_R[i, e] = 1.
                    self.all_R[i, e] = 1.

                    self.all_R_senti[i, e] = self.item2sentiment_[i]

        for i, l in self.item2visual_entity.items():
            if len(l) > 1:
                self.exist_items_in_visual.add(i)
                for e in l:
                    self.visual_R[i, e] = 1.
                    self.all_R[i, e] = 1.
                    self.all_R_senti[i, e] = self.item2sentiment_[i]


        try:
            self.ia_len = np.sum([len(item) if isinstance(item, list) else 1 for item in self.item2entity.values()])
            self.hypergraph_ui_a = sp.load_npz(origin_file + '/hypergraph_ui_a.npz').todok()
            self.hypergraph_ia_u = sp.load_npz(origin_file + '/hypergraph_ia_u.npz').todok()
        except:
            # conduct [(use, item), at_list]
            self.hypergraph_ui_a = sp.dok_matrix((len(self.exit_user_item_pair_l), self.n_users + self.n_items + self.n_entity), dtype=np.float32)
            for i in range(len(self.exit_user_item_pair_l)):
                user_id, item_id = self.exit_user_item_pair_l[i]
                # user and item
                self.hypergraph_ui_a[i, user_id] = 1.
                self.hypergraph_ui_a[i, self.n_users + item_id] = 1.

                for at in self.item2entity[item_id]: #attribute list.
                    self.hypergraph_ui_a[i, self.n_users + self.n_items + at] = 1.
            
            # pdb.set_trace()
            self.ia_len = np.sum([len(item) if isinstance(item, list) else 1 for item in self.item2entity.values()])
            # self.ia_len = sum([len(item[1]) for item in self.item2entity.items()])
            self.hypergraph_ia_u = sp.dok_matrix((self.ia_len, self.n_users + self.n_items + self.n_entity), dtype=np.float32)
            idx = 0
            for item in self.item2entity.items():
                for entity in item[1]:
                    self.hypergraph_ia_u[idx, self.n_users + item[0]] = 1.
                    self.hypergraph_ia_u[idx, self.n_users + self.n_items + entity] = 1.
                    for user in self.item2user_l[item[0]]:
                        self.hypergraph_ia_u[idx, user] = 1.
                    idx += 1
            origin_file = self.path + '/origin'
            sp.save_npz(origin_file + "/hypergraph_ui_a.npz", self.hypergraph_ui_a.tocsr())
            sp.save_npz(origin_file + "/hypergraph_ia_u.npz", self.hypergraph_ia_u.tocsr())

        print("u_i pairs: {}, i_a pairs: {}".format(len(self.exit_user_item_pair_l), self.ia_len))
        
        self.exist_items_in_entity = self.exist_items_in_title & self.exist_items_in_review & self.exist_items_in_visual
        self.R = self.R.tocsr()
        self.R_senti = self.R_senti.tocsr()
        self.title_R = self.title_R.tocsr()
        self.review_R = self.review_R.tocsr()
        self.visual_R = self.visual_R.tocsr()
        self.all_R = self.all_R.tocsr()
        self.all_R_senti = self.all_R_senti.tocsr()
        self.hypergraph_ui_a = self.hypergraph_ui_a.tocsr()
        self.hypergraph_ia_u = self.hypergraph_ia_u.tocsr()

        self.coo_R = self.R.tocoo()
        self.coo_R_senti = self.R_senti.tocoo()
        self.coo_title_R = self.title_R.tocoo()
        self.coo_review_R = self.review_R.tocoo()
        self.coo_visual_R = self.visual_R.tocoo()
        self.coo_all_R = self.all_R.tocoo()
        self.coo_all_R_senti = self.all_R_senti.tocoo()
        self.coo_hypergraph_ui_a = self.hypergraph_ui_a.tocoo()
        self.coo_hypergraph_ia_u = self.hypergraph_ia_u.tocoo()

    def get_adj_mat(self):
        origin_file = self.path + '/origin'
        all_file = self.path + '/all'
        try:
            t1 = time.time()
            if not os.path.exists(origin_file):
                os.mkdir(origin_file)
                os.mkdir(all_file)
            
            norm_adj_mat = sp.load_npz(origin_file + '/adj_mat_norm.npz')
            entity_norm_adj_mat = sp.load_npz(origin_file + '/entity_adj_mat_norm.npz')
            adj_mat_item2entity = sp.load_npz(origin_file + '/adj_mat_item2entity.npz')
            adj_mat_item2entity_symmetric = sp.load_npz(origin_file + '/adj_mat_item2entity_symmetric.npz')
            adj_mat_hypergraph_ui_a = sp.load_npz(origin_file + '/adj_mat_hypergraph_ui_a.npz')
            adj_mat_hypergraph_ia_u = sp.load_npz(origin_file + '/adj_mat_hypergraph_ia_u.npz')

            print('already load adj_t matrix', norm_adj_mat.shape, entity_norm_adj_mat.shape, time.time() - t1)

        except Exception:
            norm_adj_mat, norm_entity_adj_mat, adj_mat_item2entity, adj_mat_item2entity_symmetric, adj_mat_hypergraph_ui_a, adj_mat_hypergraph_ia_u = self.create_adj_mat()
            sp.save_npz(origin_file + '/adj_mat_norm.npz', norm_adj_mat)
            sp.save_npz(origin_file + '/norm_entity_adj_mat.npz', norm_entity_adj_mat)
            sp.save_npz(origin_file + '/adj_mat_item2entity.npz', adj_mat_item2entity)
            sp.save_npz(origin_file + '/adj_mat_item2entity_symmetric.npz', adj_mat_item2entity_symmetric)
            sp.save_npz(origin_file + '/adj_mat_hypergraph_ui_a.npz', adj_mat_hypergraph_ui_a)
            sp.save_npz(origin_file + '/adj_mat_hypergraph_ia_u.npz', adj_mat_hypergraph_ia_u)
    

        return norm_adj_mat, norm_entity_adj_mat, adj_mat_item2entity, adj_mat_item2entity_symmetric, adj_mat_hypergraph_ui_a, adj_mat_hypergraph_ia_u


    def create_adj_mat(self):
        t1 = time.time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        # item-->(users + entities)
        adj_mat_entity = sp.dok_matrix((self.n_items, self.n_users + self.n_entity), dtype=np.float32)
        adj_mat_entity = adj_mat_entity.tolil()

        adj_mat_item2entity = sp.dok_matrix((self.n_items, self.n_entity), dtype=np.float32)
        adj_mat_item2entity = adj_mat_item2entity.tolil()

        adj_mat_item2entity_symmetric = sp.dok_matrix((self.n_items + self.n_entity, self.n_items + self.n_entity), dtype=np.float32)
        adj_mat_item2entity_symmetric = adj_mat_item2entity_symmetric.tolil()

        adj_mat_hypergraph_ui_a = sp.dok_matrix((len(self.exit_user_item_pair_l), self.n_users + self.n_items + self.n_entity), dtype=np.float32)
        adj_mat_hypergraph_ui_a = adj_mat_hypergraph_ui_a.tolil()


        adj_mat_hypergraph_ia_u = sp.dok_matrix((self.ia_len, self.n_users + self.n_items + self.n_entity), dtype=np.float32)
        adj_mat_hypergraph_ia_u = adj_mat_hypergraph_ia_u.tolil()
        

        R = self.R.tolil() #(user, item)
        all_R = self.all_R.tolil() # (item, entity)


        adj_mat[:self.n_users, self.n_users: self.n_users + self.n_items] = R # user-item
        adj_mat[self.n_users: self.n_users + self.n_items, :self.n_users] = R.T

        adj_mat_entity[:, :self.n_users] = R.T # (items, users + entities)
        adj_mat_entity[:, self.n_users:] = all_R

        adj_mat_item2entity = all_R

        adj_mat_item2entity_symmetric[:self.n_items, self.n_items:] = all_R
        adj_mat_item2entity_symmetric[self.n_items:, :self.n_items] = all_R.T

        adj_mat_hypergraph_ui_a = self.hypergraph_ui_a

        adj_mat_hypergraph_ia_u = self.hypergraph_ia_u

        adj_mat = adj_mat.todok()
        adj_mat_entity = adj_mat_entity.todok()
        adj_mat_item2entity = adj_mat_item2entity.todok()
        adj_mat_item2entity_symmetric = adj_mat_item2entity_symmetric.todok()
        adj_mat_hypergraph_ui_a = adj_mat_hypergraph_ui_a.todok()
        adj_mat_hypergraph_ia_u = adj_mat_hypergraph_ia_u.todok()

        print('already create adjacency matrix', adj_mat.shape, time.time() - t1)
        t2 = time.time()

        def normalized_adj_symetric(adj, d1, d2):
            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, d1).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            d_inv_sqrt_last = np.power(rowsum, d2).flatten()
            d_inv_sqrt_last[np.isinf(d_inv_sqrt_last)] = 0.
            d_mat_inv_sqrt_last = sp.diags(d_inv_sqrt_last)

            return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt_last).tocoo()

        def row_level_normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            np.seterr(divide='ignore')
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        def normalized_adj_unsymmetric(adj):
            row_n, col_n = adj.shape[0], adj.shape[1]
            coomat = sp.coo_matrix(adj)
            indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
            data = coomat.data.astype(np.float32)
            rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(adj, axis=1) + 1e-8) + 1e-8)))
            colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(adj, axis=0) + 1e-8) + 1e-8)))
            for i in range(len(adj)):
                row = indices[i, 0]
                col = indices[i, 1]
                data[i] = data[i] * rowD[row] * colD[col]
            # pdb.set_trace()
            norm_adj = sp.coo_matrix((data, (indices[:,0], indices[:,1])), shape=(row_n, col_n))
            return norm_adj

        norm_adj_mat = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]), - 0.5, -0.5)
        norm_adj_mat = norm_adj_mat.tocsr()

        # adj_mat_entity = normalized_adj_unsymmetric(adj_mat_entity)
        adj_mat_entity = row_level_normalize(adj_mat_entity)
        adj_mat_entity = adj_mat_entity.tocsr()

        # adj_mat_item2entity = normalized_adj_unsymmetric(adj_mat_item2entity)
        adj_mat_item2entity = row_level_normalize(adj_mat_item2entity)
        adj_mat_item2entity = adj_mat_item2entity.tocsr()

        adj_mat_item2entity_symmetric = normalized_adj_symetric(adj_mat_item2entity_symmetric + sp.eye(adj_mat_item2entity_symmetric.shape[0]), -0.5, -0.5)
        adj_mat_item2entity_symmetric = adj_mat_item2entity_symmetric.tocsr()

        # adj_mat_hypergraph_ui_a = normalized_adj_unsymmetric(adj_mat_hypergraph_ui_a)
        adj_mat_hypergraph_ui_a = row_level_normalize(adj_mat_hypergraph_ui_a) # 替换成row-level normalize的效果非常棒;
        adj_mat_hypergraph_ui_a = adj_mat_hypergraph_ui_a.tocsr()

        adj_mat_hypergraph_ia_u = row_level_normalize(adj_mat_hypergraph_ia_u) # 替换成row-level normalize的效果非常棒;
        adj_mat_hypergraph_ia_u = adj_mat_hypergraph_ia_u.tocsr()
        
        print('already normalize adjacency matrix', time.time() - t2)
        return norm_adj_mat, adj_mat_entity, adj_mat_item2entity, adj_mat_item2entity_symmetric, adj_mat_hypergraph_ui_a, adj_mat_hypergraph_ia_u

    def sample_u(self):
        total_users = self.exist_users
        users = rd.sample(total_users, self.batch_size)

        def sample_pos_items_for_u(u):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]
            return pos_i_id

        def sample_neg_items_for_u(u):
            pos_items = self.train_items[u]
            while True:
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in pos_items:
                    return neg_id

        pos_items, neg_items, pos_users_for_pi, neg_users_for_pi = [], [], [], []
        for u in users:
            pos_i = sample_pos_items_for_u(u)
            neg_i = sample_neg_items_for_u(u)

            pos_items.append(pos_i)
            neg_items.append(neg_i)

        return users, pos_items, neg_items

    def sample_i_all(self):
        total_items = self.item2entity.keys()
        items = rd.sample(total_items, self.batch_size)

        def sample_pos_e_for_i(i):
            pos_entities = self.item2entity[i]
            n_pos_entities = len(pos_entities)
            pos_id = np.random.randint(low=0, high=n_pos_entities, size=1)[0]
            pos_e_id = pos_entities[pos_id]
            return pos_e_id

        def sample_neg_e_for_i(i):
            pos_entities = self.item2entity[i]
            while True:
                neg_id = np.random.randint(low=0, high=self.n_entity, size=1)[0]
                if neg_id not in pos_entities:
                    return neg_id

        pos_e, neg_e = [], []
        for i in items:
            pos_i = sample_pos_e_for_i(i)
            neg_i = sample_neg_e_for_i(i)

            pos_e.append(pos_i)
            neg_e.append(neg_i)

        return items, pos_e, neg_e


    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_entity={}, n_title_entity={}, n_review_entity={}, n_visual_entity={}'.format(self.n_entity, len(self.exist_title_entity), len(self.exist_review_entity), len(self.exist_visual_entity)))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def test_data(self):
        for u, i in self.test_set.items():
            user_batch = [0] * 100
            item_batch = [0] * 100
            test_items = []
            negative_items = []
            while len(negative_items) < 99:
                h = np.random.randint(self.n_items)
                if h in self.train_items[u]:
                    continue
                negative_items.append(h)
            test_items.extend(negative_items)
            test_items.extend(i)

            for k, item in enumerate(test_items):
                user_batch[k] = u
                item_batch[k] = item

            yield user_batch, item_batch
