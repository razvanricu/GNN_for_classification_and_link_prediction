import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from app.ml.custom_layers import (
    StrictGCNConv,
    StaticGraphAttn,
    MultiHeadStaticGAT,
    SAGEConvLayer,
    StaticGraphSAGE,
)
from app.ml.graph_info import GraphInfo


class GraphSAGELinkPredictor(keras.Model):
    """Link prediction using GraphSAGE encoder + dot-product decoder."""

    def __init__(self, out_feat, node_features, edge_index_msg, dropout_rate=0.0, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.node_features = tf.constant(node_features, dtype=tf.float32)
        self.encoder = StaticGraphSAGE(
            out_feat=out_feat,
            edge_index=edge_index_msg,
            dropout_rate=dropout_rate,
            normalize=normalize,
            final_activation=None,
        )

    def call(self, query_edges, training=None):
        all_embeddings = self.encoder(self.node_features, training=training)
        src_embeddings = tf.gather(all_embeddings, query_edges[:, 0])
        dst_embeddings = tf.gather(all_embeddings, query_edges[:, 1])
        dot_product = tf.reduce_sum(src_embeddings * dst_embeddings, axis=-1)
        return tf.nn.sigmoid(dot_product)

    def get_embeddings(self, training=False):
        return self.encoder(self.node_features, training=training)


class MLPLinkPredictor(keras.Model):
    """Link prediction using MLP encoder (no graph structure) + dot-product decoder."""

    def __init__(self, node_features, hidden_dim=64, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.node_features = tf.constant(node_features, dtype=tf.float32)

        self.dense1 = layers.Dense(hidden_dim, activation="relu")
        self.dense2 = layers.Dense(hidden_dim, activation="relu")
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, query_edges, training=None):
        x = self.drop1(self.node_features, training=training)
        x = self.dense1(x)
        x = self.drop2(x, training=training)
        x = self.dense2(x)

        x = tf.math.l2_normalize(x, axis=-1)

        src_embeddings = tf.gather(x, query_edges[:, 0])
        dst_embeddings = tf.gather(x, query_edges[:, 1])

        dot_product = tf.reduce_sum(src_embeddings * dst_embeddings, axis=-1)
        return tf.nn.sigmoid(dot_product)

    def get_embeddings(self, training=False):
        x = self.drop1(self.node_features, training=training)
        x = self.dense1(x)
        x = self.drop2(x, training=training)
        x = self.dense2(x)
        return tf.math.l2_normalize(x, axis=-1)


class GCNLinkPredictor(keras.Model):
    """Link prediction using strict GCN encoder + dot-product decoder."""

    def __init__(self, in_feat, out_feat, node_features, graph_info, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.node_features = tf.constant(node_features, dtype=tf.float32)

        self.conv1 = StrictGCNConv(in_feat, out_feat, graph_info, activation="relu")
        self.drop1 = layers.Dropout(dropout_rate)
        self.drop2 = layers.Dropout(dropout_rate)

        self.conv2 = StrictGCNConv(out_feat, out_feat, graph_info, activation=None)

    def call(self, query_edges, training=None):
        x = self.drop1(self.node_features, training=training)
        x = self.conv1(x)
        x = self.drop2(x, training=training)
        all_embeddings = self.conv2(x)

        all_embeddings = tf.math.l2_normalize(all_embeddings, axis=-1)

        src_embeddings = tf.gather(all_embeddings, query_edges[:, 0])
        dst_embeddings = tf.gather(all_embeddings, query_edges[:, 1])

        dot_product = tf.reduce_sum(src_embeddings * dst_embeddings, axis=-1)
        return tf.nn.sigmoid(dot_product)

    def get_embeddings(self, training=False):
        x = self.drop1(self.node_features, training=training)
        x = self.conv1(x)
        x = self.drop2(x, training=training)
        emb = self.conv2(x)
        return tf.math.l2_normalize(emb, axis=-1)


class GATLinkPredictor(keras.Model):
    """Link prediction using GAT encoder + dot-product decoder."""

    def __init__(self, num_heads, out_feat, node_features, edge_index_msg,
                 dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.node_features = tf.constant(node_features, dtype=tf.float32)

        in_feat = node_features.shape[1]

        self.gat1 = MultiHeadStaticGAT(
            num_heads=num_heads,
            in_feat=in_feat,
            out_feat=out_feat,
            edge_index=edge_index_msg,
            dropout_rate=dropout_rate,
            activation="relu",
        )

        self.gat2 = StaticGraphAttn(
            in_feat=num_heads * out_feat,
            out_feat=out_feat,
            edge_index=edge_index_msg,
            dropout_rate=dropout_rate,
            activation=None,
        )

        self.feature_drop = layers.Dropout(dropout_rate)

    def call(self, query_edges, training=None):
        x = self.feature_drop(self.node_features, training=training)
        x = self.gat1(x, training=training)
        x = self.gat2(x, training=training)

        x = tf.math.l2_normalize(x, axis=-1)

        src_embeddings = tf.gather(x, query_edges[:, 0])
        dst_embeddings = tf.gather(x, query_edges[:, 1])

        dot_product = tf.reduce_sum(src_embeddings * dst_embeddings, axis=-1)
        return tf.nn.sigmoid(dot_product)

    def get_embeddings(self, training=False):
        x = self.feature_drop(self.node_features, training=training)
        x = self.gat1(x, training=training)
        x = self.gat2(x, training=training)
        return tf.math.l2_normalize(x, axis=-1)
