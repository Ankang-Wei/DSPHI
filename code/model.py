import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from layer import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from utils import *
from satt import *
from tensorflow.keras.layers import Layer, LeakyReLU, Dropout



class GCNModel():

    def __init__(self,placeholders,num_features,emb_dim,features_nonzero,adj_nonzero,num_r,num_l,name,act=tf.nn.relu):

        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        #self.att = tf.Variable(tf.constant([0.5,0.33,0.25]))
        self.num_r = num_r
        self.num_l = num_l
        with tf.compat.v1.variable_scope(self.name):
            self.build()
    def build(self):
        self.adj = dropout_sparse(self.adj,1-self.adjdp,self.adj_nonzero)
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden2 = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden1)

        self.emb = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden2)

        self.embed1 = GraphAttentionLayer(
            dropout = self.dropout,
            F_=self.emb_dim)(self.hidden1)

        self.embed2 = GraphAttentionLayer(
            dropout=self.dropout,
            F_=self.emb_dim)(self.hidden2)

        self.embed3 = GraphAttentionLayer(
            dropout=self.dropout,
            F_=self.emb_dim)(self.emb)

        self.embeddings = self.embed1 + self.embed2 + self.embed3
        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, num_l=self.num_l, act=tf.nn.sigmoid)(self.embeddings)





# class HGNN_ATT(tf.keras.Model):
#     def __init__(self, input_size, n_hid, output_size, dropout=0.3):
#         super(HGNN_ATT, self).__init__()
#         self.dropout = dropout
#         self.gat1 = HGATLayer(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False, concat=True)
#         self.gat2 = HGATLayer(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True, concat=False)

#     def call(self, x, H):
#         x = self.gat1(x, H)
#         x = Dropout(self.dropout)(x, training=self.training)
#         x = self.gat2(x, H)

#         return x
