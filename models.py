import itertools
from collections import OrderedDict
import tensorflow as tf


class FieldFactorizeLayer(tf.keras.layers.Layer):
    def __init__(self, features_info, emb_dim):
        """
        sequence対応のembedding layer
        """
        super(FieldFactorizeLayer, self).__init__()
        self.emb_dim = emb_dim
        self.features_info = features_info
        self.feature_to_embedding_layer = OrderedDict()

        field_names = [feature['name'] for feature in features_info]
        self.field_combinations = itertools.combinations(field_names, 2)

        for feature in self.features_info:
            for target_feature in self.features_info:
                if target_feature['name'] != feature['name']:
                    feature_name = feature['name'] + '-' + target_feature['name']
                    self.create_embedding_layer(feature['is_sequence'], feature_name, feature['dim'])

    def create_embedding_layer(self, is_sequence, feature_name, dim):
        initializer = tf.keras.initializers.RandomNormal(stddev=0.01, seed=None)
        if is_sequence:
            # sequenceのembedding
            self.feature_to_embedding_layer[feature_name] = tf.keras.layers.Embedding(
                dim,
                self.emb_dim,
                mask_zero=True,
                name=f"embedding_{feature_name}",
                embeddings_initializer=initializer)
        else:
            self.feature_to_embedding_layer[feature_name] = tf.keras.layers.Embedding(
                dim,
                self.emb_dim,
                name=f"embedding_{feature_name}",
                embeddings_initializer=initializer)

    def embed_inputs(self, feature_name, is_sequence, inputs):
        embedding = self.feature_to_embedding_layer[feature_name](inputs)
        if is_sequence:
            # sequenceの場合はaverage pooling
            embedding = tf.math.reduce_mean(embedding, axis=1, keepdims=True)
        return embedding

    def call(self, inputs):
        field_fm_term = None
        for feature_input, feature in zip(inputs, self.features_info):
            for target_input, target_feature in zip(inputs, self.features_info):
                for feature_combination in self.field_combinations:
                    if feature['name'] == feature_combination[0] and target_feature['name'] == feature_combination[1]:
                        feature_name = feature_combination[0] + '-' + feature_combination[1]
                        target_feature_name = feature_combination[1] + '-' + feature_combination[0]
                        embedding = self.embed_inputs(
                            feature_name, feature['is_sequence'], feature_input)
                        target_embedding = self.embed_inputs(
                            target_feature_name, target_feature['is_sequence'], target_input)
                        dot_embedding = tf.matmul(embedding, target_embedding, transpose_b=True)
                        if not field_fm_term:
                            field_fm_term = dot_embedding
                        else:
                            field_fm_term = tf.add(field_fm_term, dot_embedding)

        return field_fm_term


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, features_info, emb_dim, name_prefix=''):
        """
        sequence対応のembedding layer
        """
        super(EmbeddingLayer, self).__init__()
        self.features_info = features_info
        self.feature_to_embedding_layer = OrderedDict()
        for feature in features_info:
            initializer = tf.keras.initializers.RandomNormal(stddev=0.01, seed=None)
            if feature['is_sequence']:
                # sequenceのembedding
                self.feature_to_embedding_layer[feature['name']] = tf.keras.layers.Embedding(
                    feature['dim'],
                    emb_dim,
                    mask_zero=True,
                    name=f"embedding_{name_prefix}{feature['name']}",
                    embeddings_initializer=initializer)
            else:
                self.feature_to_embedding_layer[feature['name']] = tf.keras.layers.Embedding(
                    feature['dim'],
                    emb_dim,
                    name=f"embedding_{name_prefix}{feature['name']}",
                    embeddings_initializer=initializer)

    def concatenate_embeddings(self, embeddings, name_prefix=''):
        if len(embeddings) >= 2:
            embeddings = tf.keras.layers.Concatenate(axis=1, name=name_prefix+'embeddings_concat')(embeddings)
        else:
            embeddings = embeddings[0]
        return embeddings

    def call(self, inputs):
        embeddings = []
        for feature_input, feature in zip(inputs, self.features_info):
            # embeddingの作成
            embedding = self.feature_to_embedding_layer[feature['name']](feature_input)
            if feature['is_sequence']:
                # sequenceの場合はaverage pooling
                embedding = tf.math.reduce_mean(embedding, axis=1, keepdims=True)
            embeddings.append(embedding)

        # concatenate
        embeddings = self.concatenate_embeddings(embeddings)
        return embeddings


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, features_info):
        super(LinearLayer, self).__init__()
        self.linear_embedding = EmbeddingLayer(features_info, 1, 'linear_')
        self.linear_layer = tf.keras.layers.Dense(1, activation='relu', name='linear_dense')

    def call(self, inputs):
        embeddings = self.linear_embedding(inputs)
        # reduce_sum → bias(ones_like)の方が正しいかも
        embeddings = tf.squeeze(embeddings, axis=2)
        output = self.linear_layer(embeddings)
        return output


class FFM(tf.keras.Model):
    def __init__(self, features_info, latent_dim=5):
        super(FFM, self).__init__()
        self.factorize_layer = FieldFactorizeLayer(features_info, features_info, latent_dim)
        self.linear_layer = LinearLayer(features_info)

    def call(self, inputs):
        linear_terms = self.linear_layer(inputs)
        factorization_terms = self.factorize_layer(inputs)
        output = tf.add(linear_terms, factorization_terms)

        return tf.keras.activations.sigmoid(output)
