import tensorflow as tf
from tensorflow.keras import layers


class StrictGCNConv(layers.Layer):
    """Single GCN convolution layer with symmetric normalization."""

    def __init__(self, in_feat, out_feat, graph_info, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info

        self.weight = self.add_weight(
            initializer="glorot_uniform",
            shape=(in_feat, out_feat),
            dtype="float32",
            name="gcn_weight",
        )
        self.activation = layers.Activation(activation) if activation else None

    def call(self, features):
        h = tf.matmul(features, self.weight)

        source_nodes = tf.convert_to_tensor(self.graph_info.edges[0], dtype=tf.int32)
        target_nodes = tf.convert_to_tensor(self.graph_info.edges[1], dtype=tf.int32)
        num_nodes = tf.cast(self.graph_info.num_nodes, tf.int32)

        # Add self-loops
        self_loops = tf.range(num_nodes, dtype=tf.int32)
        source_nodes = tf.concat([source_nodes, self_loops], axis=0)
        target_nodes = tf.concat([target_nodes, self_loops], axis=0)

        # Symmetric normalization
        ones = tf.ones_like(source_nodes, dtype=tf.float32)
        deg = tf.math.unsorted_segment_sum(ones, target_nodes, num_segments=num_nodes)

        deg_inv_sqrt = tf.pow(deg, -0.5)
        deg_inv_sqrt = tf.where(tf.math.is_inf(deg_inv_sqrt), 0.0, deg_inv_sqrt)

        norm_source = tf.gather(deg_inv_sqrt, source_nodes)
        norm_target = tf.gather(deg_inv_sqrt, target_nodes)
        norm = norm_source * norm_target

        # Message passing
        h_source = tf.gather(h, source_nodes)
        h_normalized = h_source * tf.expand_dims(norm, -1)

        # Aggregation
        aggregated = tf.math.unsorted_segment_sum(
            h_normalized, target_nodes, num_segments=num_nodes
        )

        return self.activation(aggregated) if self.activation else aggregated

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_feat": self.in_feat,
            "out_feat": self.out_feat,
            "graph_info": self.graph_info,
            "activation": self.activation.get_config() if self.activation else None,
        })
        return config


class StrictGCNModel(layers.Layer):
    """Two-layer GCN with LayerNorm and Dropout for node classification."""

    def __init__(self, in_feat, out_feat, num_classes, graph_info, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.gcn1 = StrictGCNConv(in_feat, out_feat, graph_info, activation="relu")
        self.ln1 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(dropout)

        self.gcn2 = StrictGCNConv(out_feat, out_feat, graph_info, activation=None)
        self.ln2 = layers.LayerNormalization()
        self.drop2 = layers.Dropout(dropout)

        self.dense = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=None):
        x = self.gcn1(inputs)
        x = self.ln1(x, training=training)
        x = self.drop1(x, training=training)

        x = self.gcn2(x)
        x = self.ln2(x, training=training)
        x = self.drop2(x, training=training)

        return self.dense(x)

    def get_config(self):
        config = super().get_config()
        return config


class StaticGraphAttn(layers.Layer):
    """Single-head GAT attention layer."""

    def __init__(self, in_feat, out_feat, edge_index, dropout_rate=0.0, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.edge_index = edge_index
        self.dropout_rate = dropout_rate

        self.W = self.add_weight(
            shape=(in_feat, out_feat), initializer="glorot_uniform", trainable=True, name="W"
        )
        self.a = self.add_weight(
            shape=(2 * out_feat, 1), initializer="glorot_uniform", trainable=True, name="a"
        )

        self.leaky = layers.LeakyReLU(0.2)
        self.attn_dropout = layers.Dropout(dropout_rate)
        self.activation = layers.Activation(activation) if activation else None

    def call(self, features, training=None):
        src = self.edge_index[0]
        dst = self.edge_index[1]
        N = tf.shape(features)[0]

        h = tf.matmul(features, self.W)

        h_i = tf.gather(h, dst)
        h_j = tf.gather(h, src)

        e = tf.squeeze(tf.matmul(tf.concat([h_i, h_j], axis=-1), self.a), axis=-1)
        e = self.leaky(e)

        # Stable softmax
        max_e = tf.math.unsorted_segment_max(e, dst, N)
        max_e_edges = tf.gather(max_e, dst)
        exp_e = tf.exp(e - max_e_edges)
        sum_exp = tf.math.unsorted_segment_sum(exp_e, dst, N)
        alpha = exp_e / (tf.gather(sum_exp, dst) + 1e-9)

        alpha = self.attn_dropout(alpha, training=training)

        messages = tf.expand_dims(alpha, -1) * h_j
        out = tf.math.unsorted_segment_sum(messages, dst, N)

        return self.activation(out) if self.activation else out

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_feat": self.in_feat,
            "out_feat": self.out_feat,
            "edge_index": self.edge_index,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation.get_config() if self.activation else None,
        })
        return config


class MultiHeadStaticGAT(layers.Layer):
    """Multi-head GAT that concatenates head outputs."""

    def __init__(self, num_heads, in_feat, out_feat, edge_index, dropout_rate=0.0, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.edge_index = edge_index
        self.dropout_rate = dropout_rate

        self.heads = [
            StaticGraphAttn(in_feat, out_feat, edge_index, dropout_rate=dropout_rate, activation=activation)
            for _ in range(num_heads)
        ]
        self.feature_dropout = layers.Dropout(dropout_rate)

    def call(self, features, training=None):
        head_outputs = [head(features, training=training) for head in self.heads]
        x = tf.concat(head_outputs, axis=-1)
        x = self.feature_dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "in_feat": self.in_feat,
            "out_feat": self.out_feat,
            "edge_index": self.edge_index,
            "dropout_rate": self.dropout_rate,
        })
        return config


class SAGEConvLayer(layers.Layer):
    """Single GraphSAGE convolution (mean aggregation)."""

    def __init__(self, out_feat, edge_index, normalize=True, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.out_feat = out_feat
        self.edge_index = edge_index
        self.normalize = normalize

        self.W = layers.Dense(out_feat, use_bias=False, kernel_initializer="glorot_uniform")
        self.activation = layers.Activation(activation) if activation else None

    def call(self, features, training=None):
        src = self.edge_index[0]
        dst = self.edge_index[1]
        num_nodes = tf.shape(features)[0]

        neighbor_features = tf.gather(features, src)
        aggregated_neighbors = tf.math.unsorted_segment_mean(
            neighbor_features, dst, num_segments=num_nodes
        )

        combined = tf.concat([features, aggregated_neighbors], axis=-1)
        out = self.W(combined)

        if self.activation:
            out = self.activation(out)
        if self.normalize:
            out = tf.math.l2_normalize(out, axis=-1)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_feat": self.out_feat,
            "edge_index": self.edge_index,
            "normalize": self.normalize,
        })
        return config


class StaticGraphSAGE(layers.Layer):
    """Two-layer GraphSAGE with dropout."""

    def __init__(self, out_feat, edge_index, dropout_rate=0.0, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.out_feat_val = out_feat
        self.edge_index_val = edge_index
        self.dropout_rate_val = dropout_rate
        self.normalize_val = normalize

        self.conv1 = SAGEConvLayer(out_feat, edge_index, normalize=normalize, activation="relu")
        self.conv2 = SAGEConvLayer(out_feat, edge_index, normalize=normalize, activation="relu")

        self.drop1 = layers.Dropout(dropout_rate)
        self.drop2 = layers.Dropout(dropout_rate)

    def call(self, features, training=None):
        x = self.drop1(features, training=training)
        x = self.conv1(x, training=training)
        x = self.drop2(x, training=training)
        x = self.conv2(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_feat": self.out_feat_val,
            "edge_index": self.edge_index_val,
            "dropout_rate": self.dropout_rate_val,
            "normalize": self.normalize_val,
        })
        return config
