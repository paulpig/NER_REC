# encoding:utf-8
import time
import tensorflow as tf
import os
import sys
from load_data_hypergraph_V2 import Data
import numpy as np
import math
import multiprocessing
import heapq
import random as rd
from tensorflow.contrib.layers import xavier_initializer
import pdb


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_name = 'MEGCF'
data_path = '../Data/'

# Choosing a specific dataset to train and test MEGCF.
# dataset = 'Art'
dataset = 'amazon-beauty'

if dataset == 'amazon-beauty':
    # Key parameters for amazon-beauty.
    # n_layers = 3
    n_layers = 3
    decay_1 = 1e-2
    # decay_1 = 0.0001
    decay_2 = 1e-3
    alpha = 0.1
elif dataset == 'Art':
    # Key parameters for Art.
    n_layers = 5
    decay_1 = 1e-2
    decay_2 = 1e-3
    alpha = 0.1
else:
    # Key parameters for Taobao.
    n_layers = 5
    decay_1 = 1e-3
    decay_2 = 1e-4
    alpha = 0.2

batch_size = 2048
lr = 0.001
embed_size = 64
epoch = 1000

data_generator = Data(path='../Data/' + dataset, batch_size=batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = batch_size
Ks = np.arange(1, 21)


"""
*********************************************************
Test function
"""
def test_one_user(x):
    u, rating = x[1], x[0]

    training_items = data_generator.train_items[u]

    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = rd.sample(list(all_items - set(training_items) - set(user_pos_test)), 99) + user_pos_test

    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    precision, recall, ndcg, hit_ratio = [], [], [], []

    def hit_at_k(r, k):
        r = np.array(r)[:k]
        if np.sum(r) > 0:
            return 1.
        else:
            return 0.

    def ndcg_at_k(r, k):
        r = np.array(r)[:k]

        if np.sum(r) > 0:
            return math.log(2) / math.log(np.where(r == 1)[0] + 2)
        else:
            return 0.

    for K in Ks:
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))

    return {'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}

def test(sess, model, users, items, batch_size, cores):

    result = {'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(cores)

    u_batch_size = batch_size * 2

    n_test_users = len(users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):

        start = u_batch_id * u_batch_size

        end = (u_batch_id + 1) * u_batch_size

        user_batch = users[start: end]

        item_batch = items

        rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                    model.pos_items: item_batch})

        user_batch_rating_uid = zip(rate_batch, user_batch)

        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        count += len(batch_result)

        for re in batch_result:
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users

    assert count == n_test_users
    pool.close()
    return result


"""
*********************************************************
Construct MEGCF model
"""
class Model(object):

    def __init__(self, data_config):

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entity = data_config['n_entity']

        # self.n_fold = 100
        self.n_fold = 500

        self.norm_adj = data_config['norm_adj']
        self.entity_norm_adj = data_config['all_norm_adj'] # (item, user + entities)
        self.adj_mat_item2entity = config['adj_mat_item2entity']
        self.adj_mat_hypergraph_ui_a = config['adj_mat_hypergraph_ui_a']

        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = data_config['lr']

        self.emb_dim = data_config['embed_size']
        self.batch_size = data_config['batch_size']

        self.n_layers = data_config['n_layers']

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.items_all = tf.placeholder(tf.int32, shape=(None,))
        self.pos_e_all = tf.placeholder(tf.int32, shape=(None,))
        self.neg_e_all = tf.placeholder(tf.int32, shape=(None,))

        self.weights = self._init_weights()

        # v1: layer layer, no_norm
        self.all_embeddings = self._create_norm_embed()
        # v2: reduce_sum after l2 normalizing.
        # self.all_embeddings = [self._create_norm_embed_norm()] 

        # self.all_embeddings_entity, _ = self._create_norm_embed_entity()
        self.all_embeddings_entity = [self._create_norm_embed_hypergraph_ui_a()] #(n_users + n_items + n_entity, dim)
        
        # self.ua_embeddings_4, self.ia_embeddings_4 = self.all_embeddings[self.n_layers - 1] # last layer
        self.ua_embeddings_4, self.ia_embeddings_4 = tf.split(self.all_embeddings[-1], [self.n_users, self.n_items], 0) # sum layer
        self.u_g_embeddings_4 = tf.nn.embedding_lookup(self.ua_embeddings_4, self.users)
        self.pos_i_g_embeddings_4 = tf.nn.embedding_lookup(self.ia_embeddings_4, self.pos_items)
        self.neg_i_g_embeddings_4 = tf.nn.embedding_lookup(self.ia_embeddings_4, self.neg_items)

        self.ua_embeddings_4_all, self.ia_embeddings_4_all, self.entity_embedding_4_all = tf.split(self.all_embeddings_entity[-1], [self.n_users, self.n_items, self.n_entity], 0) # sum layer
        self.u_g_embeddings_4_all = tf.nn.embedding_lookup(self.ua_embeddings_4_all, self.users)
        self.pos_i_g_embeddings_4_all = tf.nn.embedding_lookup(self.ia_embeddings_4_all, self.pos_items)
        self.neg_i_g_embeddings_4_all = tf.nn.embedding_lookup(self.ia_embeddings_4_all, self.neg_items)

        # for inference, single loss;
        # self.batch_ratings = tf.matmul(self.u_g_embeddings_4, self.pos_i_g_embeddings_4, transpose_a=False, transpose_b=True)
        # self.loss, self.mf_loss, self.loss_all, self.emb_loss = self.create_bpr_loss()

        # for inference, two losses;
        self.batch_ratings = tf.matmul(self.u_g_embeddings_4, self.pos_i_g_embeddings_4, transpose_a=False, transpose_b=True) +\
                tf.matmul(self.u_g_embeddings_4_all, self.pos_i_g_embeddings_4_all, transpose_a=False, transpose_b=True)
        
        self.loss, self.mf_loss, self.emb_loss = self.create_bpr_loss_two_bpr_loss()

        # self.u_cl_loss = self.instance_level_cl_loss(self.u_g_embeddings_4, self.u_g_embeddings_4_all)
        # self.i_pos_cl_loss = self.instance_level_cl_loss(self.pos_i_g_embeddings_4, self.pos_i_g_embeddings_4_all)
        # self.i_neg_cl_loss = self.instance_level_cl_loss(self.neg_i_g_embeddings_4, self.neg_i_g_embeddings_4_all)

        # # cl_w = 5.e-4 # cl_loss: 12700;
        # # cl_w = 1.e-4 # cl_loss: 12700;
        # cl_w = 5.e-5 # cl_loss: 12700;
        # self.loss = self.loss + cl_w * (self.u_cl_loss + self.i_pos_cl_loss)

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):

        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')
        all_weights['e_embedding'] = tf.Variable(initializer([self.n_entity, self.emb_dim]),
                                                 name='e_embedding')
        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items + self.n_entity) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items + self.n_entity
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_right(self, X):
        """
        X: (a, b)
        """
        row, col = X.shape
        A_fold_hat = []

        fold_len = row // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = row
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_norm_embed(self):
        A_fold_hat = self._split_A_hat_right(self.norm_adj)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        # all_embeddings = {}
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            u_g_embeddings_4, i_g_embeddings_4 = tf.split(ego_embeddings, [self.n_users, self.n_items], 0)
            # all_embeddings[k] = [u_g_embeddings_4, i_g_embeddings_4]
            all_embeddings.append(tf.concat([u_g_embeddings_4, i_g_embeddings_4], axis=0))
        # sum
        # all_embeddings = tf.reduce_sum(tf.stack(all_embeddings), axis=0) #
        # mean
        # all_embeddings = tf.reduce_mean(tf.stack(all_embeddings), axis=0) #
        return all_embeddings

    def _create_norm_embed_norm(self):
        A_fold_hat = self._split_A_hat_right(self.norm_adj)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        # all_embeddings = {}
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            # add norm
            side_embeddings = tf.math.l2_normalize(side_embeddings, axis=1)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)
        
        # sum
        all_embeddings = tf.reduce_sum(all_embeddings, axis=0) #
        # mean
        # all_embeddings = tf.reduce_mean(tf.stack(all_embeddings), axis=0) #
        return all_embeddings

    def edgeDropout(self, mat):
        def dropOneMat(mat):
            indices = mat.indices
            values = mat.values
            shape = mat.dense_shape
            # newVals = tf.to_float(tf.sign(tf.nn.dropout(values, self.keepRate)))
            newVals = tf.nn.dropout(values, self.keepRate)
            return tf.sparse.SparseTensor(indices, newVals, shape)
        return dropOneMat(mat)


    def FC(self, inp, outDim, name=None, useBias=False, activation=None, dropout=None, layer_idx=0):
        inDim = inp.get_shape()[1]
        name = name + "_{}".format(layer_idx)

        W = tf.get_variable(name=name, dtype=tf.float32, shape=[inDim, outDim], initializer=xavier_initializer(dtype=tf.float32), trainable=True)

        if dropout != None:
            ret = tf.nn.dropout(inp, rate=dropout) @ W
        else:
            ret = inp @ W
        if useBias:
            ret = ret + tf.get_variable(name=name + "_bias", dtype=tf.float32, shape=[outDim], initializer=xavier_initializer(dtype=tf.float32), trainable=True)
        if activation != None:
            ret = tf.nn.leaky_relu(ret)
        return ret
    
    def _create_norm_embed_entity(self):
        A_fold_hat = self._split_A_hat_right(self.entity_norm_adj) #(item, user + entity)
        tp_A_fold_hat = self._split_A_hat_right(self.entity_norm_adj.T)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['e_embedding']], axis=0)

        all_embeddings = [ego_embeddings]
        item_embeddings = []
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings)) 
            side_embeddings = tf.concat(temp_embed, 0) #(item, dim)            # add FC
            side_embeddings = self.FC(side_embeddings, self.emb_dim, name="fc_1", activation='leaky_relu', layer_idx=k) #(item, dim)
            side_embeddings = self.FC(side_embeddings, self.emb_dim, name="fc_2", activation='leaky_relu', layer_idx=k) #(item, dim)
            side_embeddings = self.FC(side_embeddings, self.emb_dim, name="fc_3", activation='leaky_relu', layer_idx=k) #(item, dim)
            item_embeddings.append(side_embeddings)

            temp_embed_2  = []
            for f in range(self.n_fold):
                temp_embed_2.append(tf.sparse_tensor_dense_matmul(tp_A_fold_hat[f], side_embeddings)) 
            ego_tmp_embeddings = tf.concat(temp_embed_2, 0) #(user + entity, dim)
            all_embeddings.append(ego_tmp_embeddings)
            ego_embeddings = ego_tmp_embeddings
        
        return all_embeddings, item_embeddings


    """
    参考的代码; https://github.com/ziruizhu/SHGCN/blob/master/model/SHGCN.py
    """
    # def forward(self, E, CE_adj, EC_adj, RC_adj, UU_idx):
    #     c_emb = self.C_transformer(torch.sparse.mm(CE_adj, E))
    #     r_emb = self.R_transformer(torch.sparse.mm(RC_adj, c_emb))

    #     R_rating = self.rating_func(r_emb).squeeze(dim=1)
    #     uu_message = SpecialSpmm(UU_idx, R_rating, torch.Size([self.num_user, self.num_user]), E[:self.num_user])
    #     uu_message = F.leaky_relu(uu_message)
    #     ec_message = F.leaky_relu(torch.sparse.mm(EC_adj, c_emb))
    #     E = ec_message
    #     E[:self.num_user] = E[:self.num_user] + uu_message
    #     return 

    def _create_norm_embed_hypergraph_ui_a(self):
        A_fold_hat = self._split_A_hat_right(self.adj_mat_hypergraph_ui_a) #(item, user + entity)
        tp_A_fold_hat = self._split_A_hat_right(self.adj_mat_hypergraph_ui_a.T)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding'], self.weights['e_embedding']], axis=0)

        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            # passing to hyepredge
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings)) 
            side_embeddings = tf.concat(temp_embed, 0) #(item, dim)            # add FC
            side_embeddings = self.FC(side_embeddings, self.emb_dim, name="fc_1", activation='leaky_relu', layer_idx=k) #(item, dim)
            # side_embeddings = self.FC(side_embeddings, self.emb_dim, name="fc_2", activation='leaky_relu', layer_idx=k) #(item, dim)
            # side_embeddings = self.FC(side_embeddings, self.emb_dim, name="fc_3", activation='leaky_relu', layer_idx=k) #(item, dim)

            # pdb.set_trace()
            #passing to nodes
            temp_embed_2  = []
            for f in range(self.n_fold):
                temp_embed_2.append(tf.sparse_tensor_dense_matmul(tp_A_fold_hat[f], side_embeddings)) 
            ego_tmp_embeddings = tf.concat(temp_embed_2, 0) #(user + entity, dim)

            # add other hypergraph, 计算用户对每个属性的喜好程度, TODO.

            # add norm
            ego_tmp_embeddings = tf.math.l2_normalize(ego_tmp_embeddings, axis=1)
            all_embeddings.append(ego_tmp_embeddings)
            ego_embeddings = ego_tmp_embeddings
        
        # sum
        all_embeddings = tf.reduce_sum(all_embeddings, axis=0)
        
        return all_embeddings

    def instance_level_cl_loss(self, user_embeddings, edge_embeddings):
        """
        instance level contrastive learning;
        """
        def row_shuffle(embedding):
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))
        def row_column_shuffle(embedding):
            corrupted_embedding = tf.transpose(tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0])))) #(user, dim)
            corrupted_embedding = tf.gather(corrupted_embedding, tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding
        def score(x1,x2):
            return tf.reduce_sum(tf.multiply(x1,x2),1)
        
        # l2 normilize
        user_embeddings = tf.math.l2_normalize(user_embeddings, axis=1)
        edge_embeddings = tf.math.l2_normalize(edge_embeddings, axis=1)

        # loss MIM
        pos = score(user_embeddings,edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings),edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings),user_embeddings)
        local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos-neg1))-tf.log(tf.sigmoid(neg1-neg2)))
        #Global MIM
        graph = tf.reduce_mean(edge_embeddings,0)
        pos = score(edge_embeddings,graph)
        neg1 = score(row_column_shuffle(edge_embeddings),graph)
        global_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos-neg1)))
        return local_loss + global_loss


    def create_bpr_loss_two_bpr_loss(self):

        pos_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4, self.pos_i_g_embeddings_4), axis=1)
        neg_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4, self.neg_i_g_embeddings_4), axis=1)

        pos_scores_4_all = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4_all, self.pos_i_g_embeddings_4_all),
                                         axis=1)
        neg_scores_4_all = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4_all, self.neg_i_g_embeddings_4_all),
                                         axis=1)

        regularizer_mf = tf.nn.l2_loss(self.u_g_embeddings_4) + tf.nn.l2_loss(self.pos_i_g_embeddings_4) + \
                         tf.nn.l2_loss(self.neg_i_g_embeddings_4)

        regularizer_all = tf.nn.l2_loss(self.u_g_embeddings_4_all) + tf.nn.l2_loss(self.pos_i_g_embeddings_4_all) + \
                          tf.nn.l2_loss(self.neg_i_g_embeddings_4_all)

        emb_loss = (decay_1 * regularizer_mf + decay_2 * regularizer_all) / self.batch_size
        # emb_loss = (decay_1 * regularizer_mf) / self.batch_size

        mf_loss_4 = tf.reduce_mean(tf.nn.softplus(-(pos_scores_4 - neg_scores_4)))

        loss_4_all = tf.reduce_mean(tf.nn.softplus(-(pos_scores_4_all - neg_scores_4_all)))

        loss = mf_loss_4 + emb_loss + loss_4_all
        # loss = mf_loss_4 + emb_loss

        return loss, mf_loss_4, emb_loss

    def create_bpr_loss_single_obj(self):

        pos_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4, self.pos_i_g_embeddings_4), axis=1)
        neg_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4, self.neg_i_g_embeddings_4), axis=1)

        regularizer_mf = tf.nn.l2_loss(self.u_g_embeddings_4) + tf.nn.l2_loss(self.pos_i_g_embeddings_4) + \
                         tf.nn.l2_loss(self.neg_i_g_embeddings_4)

        emb_loss = (decay_1 * regularizer_mf) / self.batch_size

        mf_loss_4 = tf.reduce_mean(tf.nn.softplus(-(pos_scores_4 - neg_scores_4)))

        loss = mf_loss_4 + emb_loss

        return loss, mf_loss_4, emb_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)


if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    if not os.path.exists('Log/'):
        os.mkdir('Log/')
    # file = open('Log/ours-{}-result-{}-decay={}-layer=3.txt'.format(time.time(), dataset, decay_1, decay_2), 'a')

    cores = multiprocessing.cpu_count() // 2
    Ks = np.arange(1, 21)

    data_generator.print_statistics()
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_entity'] = data_generator.n_entity
    config['n_layers'] = n_layers
    config['embed_size'] = embed_size
    config['lr'] = lr
    config['batch_size'] = batch_size


    norm, norm_entity, adj_mat_item2entity, _, adj_mat_hypergraph_ui_a = data_generator.get_adj_mat() # 3, 4 mean that the matrix with different normalization weight.

    config['norm_adj'] = norm
    config['all_norm_adj'] = norm_entity
    config['adj_mat_item2entity'] = adj_mat_item2entity
    config['adj_mat_hypergraph_ui_a'] = adj_mat_hypergraph_ui_a

    print('shape of adjacency', norm.shape, norm_entity.shape)

    t0 = time.time()

    model = Model(data_config=config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    max_recall, max_precision, max_ndcg, max_hr = 0., 0., 0., 0.
    max_ndcg_5, max_ndcg_10, max_ndcg_20, max_hr_5, max_hr_10, max_hr_20 = 0., 0., 0., 0., 0., 0.
    max_epoch = 0
    earlystop_iter = 0

    best_score = 0
    best_result = {}
    all_result = {}

    for epoch in range(epoch):
        t1 = time.time()
        loss, mf_loss, title_loss, review_loss, visual_loss, all_loss, emb_loss, kg_loss, cl_loss = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        n_batch = data_generator.n_train // batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample_u()
            items_all, pos_e_all, neg_e_all = data_generator.sample_i_all()

            # _, batch_loss, batch_mf_loss, batch_emb_loss, batch_u_cl_loss, batch_i_pos_cl_loss, batch_i_neg_cl_loss = sess.run(
            #     [model.opt, model.loss, model.mf_loss, model.emb_loss, model.u_cl_loss, model.i_pos_cl_loss, model.i_neg_cl_loss],
            _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                [model.opt, model.loss, model.mf_loss, model.emb_loss],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items,
                           model.items_all: items_all,
                           model.pos_e_all: pos_e_all,
                           model.neg_e_all: neg_e_all
                           })
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            # cl_loss += (batch_u_cl_loss + batch_i_pos_cl_loss + batch_i_neg_cl_loss)
            # print("total_loss: {}, u_cl_loss: {}, pos_i_cl_loss: {}, neg_i_cl_loss: {}".format(batch_loss, batch_u_cl_loss, batch_i_pos_cl_loss, batch_i_neg_cl_loss))

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % 10 != 0:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                epoch, time.time() - t1, loss, mf_loss, all_loss, emb_loss)
            # print("u_cl_loss: {}, pos_i_cl_loss: {}, neg_i_cl_loss: {}".format(batch_u_cl_loss, batch_i_pos_cl_loss, batch_i_neg_cl_loss))
            # print("cl_loss: {}".format(cl_loss))
            print(perf_str)
            continue

        """
        *********************************************************
        Test.
        """
        t2 = time.time()
        users_to_test = list(data_generator.test_set.keys())

        result = test(sess, model, users_to_test, data_generator.exist_items, batch_size, cores)
        hr = result['hit_ratio']
        ndcg = result['ndcg']

        t3 = time.time()

        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f],hit@5=[%.5f],hit@10=[%.5f],hit@20=[%.5f],ndcg@5=[%.5f],ndcg@10=[%.5f],ndcg@20=[%.5f]' % \
                   (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, hr[4], hr[9], hr[19], ndcg[4], ndcg[9], ndcg[19])
        print(perf_str)

        if int(hr[9] > max_hr_10) +  int(ndcg[9] > max_ndcg_10) + int(hr[4] > max_hr_5) + int(ndcg[4] > max_ndcg_5) + int(hr[19] > max_hr_20) + int(ndcg[19] > max_ndcg_20)> 3:
            max_hr_5, max_hr_10, max_hr_20, max_ndcg_5, max_ndcg_10, max_ndcg_20, max_epoch = hr[4], hr[9], hr[19], ndcg[4], ndcg[9], ndcg[19], epoch
            print("Best Epoch: %d, hit@5=[%.5f],hit@10=[%.5f],hit@20=[%.5f],ndcg@5=[%.5f],ndcg@10=[%.5f],ndcg@20=[%.5f]" % (max_epoch, max_hr_5, max_hr_10, max_hr_20, max_ndcg_5, max_ndcg_10, max_ndcg_20))
            earlystop_iter = 0
        else:
            earlystop_iter += 1
        
        if earlystop_iter >= 10:
            print("Final Epoch: %d, hit@5=[%.5f],hit@10=[%.5f],hit@20=[%.5f],ndcg@5=[%.5f],ndcg@10=[%.5f],ndcg@20=[%.5f]" % (max_epoch, max_hr_5, max_hr_10, max_hr_20, max_ndcg_5, max_ndcg_10, max_ndcg_20))
            exit(0)
