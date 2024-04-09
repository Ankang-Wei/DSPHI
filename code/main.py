import numpy as np
import scipy.sparse as sp
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import  ops
ops.reset_default_graph()
import gc
import random
from layer import *
from metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer

def PredictScore(train_phage_host_matrix, phage_matrix, host_matrix, high_adj,  h_adj, seed, epochs, emb_dim,dp,lr,adjdp,simw,simz):
    np.random.seed(seed)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)

    # high_avg = avgnew(phage_matrix, high_adj)   #平均池化后的新节点特征
    # #high_avg = avg(train_phage_host_matrix, high_adj)
    # #high_avg = max(train_phage_host_matrix, high_adj)
    # high_adjj = cos(high_avg, phage_matrix)
    # high_adjv = np.multiply(high_adjj, high_adj) #新节点与原始节点间的连边关系

    phage_sim = np.where(phage_matrix > simv, phage_matrix, 0)
    host_sim = np.where(host_matrix > simh, host_matrix, 0)

    adj = constructHNet(train_phage_host_matrix, phage_sim, host_sim, high_adj * simw, h_adj * simz)
    adj = sp.csc_matrix(adj)
    association_nam= train_phage_host_matrix.sum()
    X = constructNet(train_phage_host_matrix, high_adj, h_adj)
    features = sparse_to_tuple(sp.csc_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_phage_host_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csc_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features':tf.compat.v1.sparse_placeholder(tf.float32),
        'adj':tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig':tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout':tf.compat.v1.placeholder_with_default(0.,shape=()),
        'adjdp':tf.compat.v1.placeholder_with_default(0.,shape=())
    }
    model = GCNModel(placeholders,num_features,emb_dim,
                     features_nonzero,adj_nonzero,train_phage_host_matrix.shape[0],train_phage_host_matrix.shape[1],name='GCNGAT')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr,num_u=train_phage_host_matrix.shape[0],num_v=train_phage_host_matrix.shape[1],association_nam=association_nam)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res

def cross_validation_experiment(phage_host_matrix, phage_matrix, host_matrix,high_adj, h_adj,seed, epochs, emb_dim, dp, lr, adjdp,simw,simz):
    #进行交叉验证
    index_matrix = np.mat(np.where(phage_host_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating met-disease...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(phage_host_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        phage_len = phage_host_matrix.shape[0]
        host_len = phage_host_matrix.shape[1]
        phage_disease_res = PredictScore(
            train_matrix, phage_matrix, host_matrix,high_adj, h_adj, seed, epochs, emb_dim, dp, lr,  adjdp,simw,simz)
        predict_y_proba = phage_disease_res.reshape(phage_len, host_len)


        # 保存每次交叉验证的 predict_y_proba 到 CSV 文件
        csv_filename = f"predict_y_proba_fold{k + 1}.csv"
        np.savetxt(csv_filename, predict_y_proba, delimiter=",")


    #添加输出结果的代码
        metric_tmp = cv_model_evaluate(
            phage_host_matrix, predict_y_proba, train_matrix)
        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    return metric

if __name__ == "__main__":
    phage_sim = np.loadtxt('data/V.csv', delimiter=',')
    host_sim = np.loadtxt('data/H.csv', delimiter=',')
    h_adj = np.loadtxt('data/high-h.csv', delimiter=',')
    high_adj = np.loadtxt('data/high3-JH-HB-adj.csv', delimiter=',')
    phage_host_matrix = np.loadtxt('data/VH.csv', delimiter=',')
    epoch = 4000
    emb_dim = 64
    lr = 0.01
    adjdp = 0.7
    dp = 0.2
    simv = 0.98
    simh = 1
    simw = 1
    simz = 0.1

    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    for i in range(circle_time):
        result += cross_validation_experiment(
            phage_host_matrix, phage_sim, host_sim, high_adj, h_adj, i, epoch, emb_dim, dp, lr, adjdp,simw,simz)
    average_result = result / circle_time
    print(average_result)
