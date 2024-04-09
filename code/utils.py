import numpy as np
#import torch
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.sparse as sp
from tensorflow.python.ops import array_ops

def weight_variable_glorot(input_dim,output_dim,name=""):
    init_range = np.sqrt(6.0/(input_dim+output_dim))
    initial = tf.compat.v1.random_uniform(
        [input_dim,output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )

    return tf.Variable(initial,name=name)


def dropout_sparse(x,keep_prob,num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor +=tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor),dtype=tf.bool)
    pre_out = tf.sparse_retain(x,dropout_mask)
    return pre_out*(1./keep_prob)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row,sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords,values,shape

def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum,-0.5).flatten())
    adj_nomalized = adj_.dot(
        degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_nomalized = adj_nomalized.tocoo()
    return  sparse_to_tuple(adj_nomalized)
#
# def constructNet(met_dis_matrix):
#     met_matrix = np.matrix(
#         np.zeros((met_dis_matrix.shape[0],met_dis_matrix.shape[0]),dtype=np.int8))
#     dis_matrix = np.matrix(
#         np.zeros((met_dis_matrix.shape[1],met_dis_matrix.shape[1]),dtype=np.int8))
#
#     mat1 = np.hstack((met_matrix,met_dis_matrix))
#     mat2 = np.hstack((met_dis_matrix.T,dis_matrix))
#     adj = np.vstack((mat1,mat2))
#     return adj

#考虑平均池化初始节点特征表示

#
# def constructNet(met_dis_matrix, high_sim):
#     met_matrix = np.matrix(
#         np.zeros((met_dis_matrix.shape[0],met_dis_matrix.shape[0]),dtype=np.int8))
#     dis_matrix = np.matrix(
#         np.zeros((met_dis_matrix.shape[1],met_dis_matrix.shape[1]),dtype=np.int8))
#
#     mat1 = np.hstack((met_matrix,met_dis_matrix))
#     mat2 = np.hstack((met_dis_matrix.T,dis_matrix))
#
#     high_met = np.matrix(
#         np.zeros((high_sim.shape[0],mat1.shape[1]),dtype=np.int8))
#
#     met3 = np.vstack((mat1,mat2,high_met))
#
#     return met3
#节点特征除了噬菌体宿主以外使用全0元素作为新节点的初始输入
# def constructNet(met_m, dis_m, high_m):
#     met_matrix = np.matrix(
#         np.zeros((met_m.shape[0],dis_m.shape[1]),dtype=np.int8))
#     dis_matrix = np.matrix(
#         np.zeros((dis_m.shape[0],met_m.shape[1]),dtype=np.int8))
#     high_matrix = np.matrix(
#         np.zeros((high_m.shape[0],dis_m.shape[1]), dtype=np.int8))

#     mat1 = np.hstack((met_m,met_matrix))
#     mat2 = np.hstack((dis_matrix,dis_m))
#     mat3 = np.hstack((high_m,high_matrix))
#     adjj = np.vstack((mat1,mat2))
#     adj = np.vstack((adjj, mat3))
#     return adj

#邻接矩阵的构建
def constructHNet(met_dis_matrix, met_matrix, dis_matrix, high_adj, h_adj, high_adx=None):
    h_v3 = np.matrix(
        np.zeros((dis_matrix.shape[0],high_adj.shape[0]),dtype=np.int8))
    h_h3 = np.matrix(
        np.zeros((met_matrix.shape[0],h_adj.shape[0]),dtype=np.int8))
    h_vh3 =np.matrix(
        np.zeros((high_adj.shape[0], h_adj.shape[0]), dtype=np.int8))
    mat1 = np.hstack((met_matrix,met_dis_matrix,high_adj.T,h_h3))
    mat2 = np.hstack((met_dis_matrix.T,dis_matrix,h_v3,h_adj.T))
    mat3 = np.hstack((high_adj,h_v3.T))
    high_v3 = np.eye(high_adj.shape[0],high_adj.shape[0], dtype=np.int8)
    mat4 = np.hstack((mat3, high_v3, h_vh3))
    high_h3 = np.eye(h_adj.shape[0], h_adj.shape[0], dtype=np.int8)
    mat5 = np.hstack((h_h3.T, h_adj, h_vh3.T,high_h3))
    mat6 = np.vstack((mat1,mat2,mat4,mat5))

    return mat6

# def constructHNet(met_dis_matrix, met_matrix, dis_matrix,high_adjv):
#     h_v3 = np.matrix(
#         np.zeros((dis_matrix.shape[0],high_adjv.shape[0]),dtype=np.int8))
#
#     mat1 = np.hstack((met_matrix,met_dis_matrix,high_adjv.T))
#     mat2 = np.hstack((met_dis_matrix.T,dis_matrix,h_v3))
#     mat3 = np.hstack((high_adjv,h_v3.T))
#     high_v3 = np.eye(high_adjv.shape[0],high_adjv.shape[0], dtype=np.int8)
#     mat4 = np.hstack((mat3, high_v3))
#     mat5 = np.vstack((mat1,mat2,mat4))
#
#     return mat5


# def constructHNet(met_dis_matrix,met_matrix,dis_matrix,high_sim,high_adj):
#     h_v3 = np.matrix(np.zeros((dis_matrix.shape[0],high_sim.shape[1]),dtype=np.int8))
#
#     mat1 = np.hstack((met_matrix,met_dis_matrix,high_adj.T))
#     mat2 = np.hstack((met_dis_matrix.T,dis_matrix,h_v3))
#     mat3 = np.hstack((high_adj,h_v3.T, high_sim))
#     mat4 = np.hstack((mat1, mat2,mat3))
#
#     return mat4




import numpy as np

def avgnew(matrix_A, matrix_B):
    # 获取 A: 特征矩阵 和 B: 新节点与原始节点的邻接矩阵（关联矩阵） 的形状信息
    n = matrix_A.shape[1]
    m = matrix_B.shape[0]

    # 创建一个NumPy数组 C 并初始化为零矩阵
    C = np.zeros((m, n), dtype=np.float32)

    # 遍历比对行
    for i in range(m):
        # 读取 B 的第 i 行，并提取非零元素的位置索引值
        row_B = matrix_B[i]
        non_zero_indices = np.where(row_B != 0)[0]

        # 提取 A 中对应的行向量
        selected_rows_A = matrix_A[non_zero_indices]

        # 计算余弦相似性
        similarity_matrix = np.dot(selected_rows_A, selected_rows_A.T)
        norms_A = np.linalg.norm(selected_rows_A, axis=1, keepdims=True)
        similarity_matrix = similarity_matrix / (norms_A @ norms_A.T)

        # 计算最相似的两个行向量的索引
        max_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        max_score = similarity_matrix[max_indices]

        # 获取最相似的两个行向量
        most_similar_row1 = selected_rows_A[max_indices[0]]
        most_similar_row2 = selected_rows_A[max_indices[1]]

        # 计算平均池化操作，得到一维的节点向量表示
        pooled_vector = (most_similar_row1 + most_similar_row2) / 2.0

        # 将该行向量填充到 C 矩阵的第 i 行
        C[i] = pooled_vector

    return C





def avg(matrix_A, matrix_B):
    # 计算 A:特征矩阵 的列数 n 和 B：关联矩阵 的行数 m
    n = matrix_A.shape[1]
    m = matrix_B.shape[0]

    # 创建零元矩阵 C
    C = np.zeros((m, n))

    # 遍历比对行
    for i in range(m):
        # 读取 B 的第 i 行，并提取非零元素的位置索引值
        row_B = matrix_B[i]
        non_zero_indices = np.where(row_B != 0)[0]

        # 提取 A 中对应的行向量
        selected_rows_A = matrix_A[non_zero_indices]

        # 计算平均池化操作，得到一维的节点向量表示
        pooled_vector = np.mean(selected_rows_A, axis=0)

        # 将该行向量填充到 C 矩阵的第 i 行
        C[i] = pooled_vector

    return C


def max(matrix_A, matrix_B):
    # 计算 A:特征矩阵 的列数 n 和 B：关联矩阵 的行数 m
    n = matrix_A.shape[1]
    m = matrix_B.shape[0]

    # 创建零元矩阵 C
    C = np.zeros((m, n))

    # 遍历比对行
    for i in range(m):
        # 读取 B 的第 i 行，并提取非零元素的位置索引值
        row_B = matrix_B[i]
        non_zero_indices = np.where(row_B != 0)[0]

        # 提取 A 中对应的行向量
        selected_rows_A = matrix_A[non_zero_indices]

        # 计算平均池化操作，得到一维的节点向量表示
        pooled_vector = np.max(selected_rows_A, axis=0)

        # 将该行向量填充到 C 矩阵的第 i 行
        C[i] = pooled_vector

    return C


def cos(matrix_A, matrix_B):
    # 明确将输入数组的数据类型设置为 float32
    matrix_A = matrix_A.astype(np.float32)
    matrix_B = matrix_B.astype(np.float32)

    # 计算矩阵A的范数
    norm_A = np.linalg.norm(matrix_A, axis=1, keepdims=True)

    # 计算矩阵B的范数
    norm_B = np.linalg.norm(matrix_B, axis=1, keepdims=True)

    # 计算矩阵A和B的内积
    dot_product = np.dot(matrix_A, matrix_B.T)

    # 计算余弦相似性
    cosine_similarity = dot_product / (np.dot(norm_A, norm_B.T))

    return cosine_similarity






def constructNet(met_dis_matrix, high_adj, h_adj):
    met_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[0], met_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[1], met_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((met_matrix, met_dis_matrix))
    mat2 = np.hstack((met_dis_matrix.T, dis_matrix))

    high_met = max(met_dis_matrix, high_adj)
    h_met = max(met_dis_matrix, h_adj)

    high_matrix = np.matrix(
        np.zeros((high_met.shape[0], met_dis_matrix.shape[0]), dtype=np.int8))
    h_matrix = np.matrix(
        np.zeros((h_met.shape[0], met_dis_matrix.shape[0]), dtype=np.int8))

    high_met2 = np.hstack((high_matrix, high_met))
    h_met2 = np.hstack((h_met,h_matrix))
    met3 = np.vstack((mat1, mat2, high_met2,h_met2 ))

    return met3