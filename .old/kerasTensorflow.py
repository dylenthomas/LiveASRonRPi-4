import re
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import layers

from glob import glob

pattern_wav_name = re.compile(r'([^/\\\.]+)')

#https://keras.io/examples/audio/transformer_asr/

class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)

        return x + positions
    
class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.conv2 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.conv3 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x
    
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(feed_forward_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epislon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  
        
        return out2

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6) 
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6) 
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6) 
        self.self_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_droppout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential([
            layers.Dense(feed_forward_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])

    def casual_attention_mask(self, batch_size, n_dset, n_src, dtype):
        i = tf.range(n_dset)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dset
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dset, n_src])
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0)

        return tf.title(mask, mult)
    
    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.set_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_droppout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm, self.ffn_dropout(ffn_out))

        return ffn_out_norm
    
class Transformer(keras.Model):
    def __init__(self,
                  num_hid=64,
                  num_head=2,
                  num_feed_forward=128,
                  source_maxlen=100,
                  target_maxlen=100,
                  num_layers_enc=4, 
                  num_layers_dec=1,
                  num_classes=10
                  ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid)

        self.encoder = keras.Sequentiial(
            [self.enc_input] + [TransformerEncoder(num_hid, num_head, num_feed_forward) for _ in range(num_layers_enc)]
        )

        # store the encoders as attributes of the transformer to be individually called
        for i in range(num_layers_dec):
            setattr(
                self,
                "dec_layer_{}".format(i),
                TransformerDecoder(num_hid, num_head, num_feed_forward)
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, "dec_layer_{}".format(i))(enc_out, y)
        
        return y
    
    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)

        return self.classifier(y)
    
    @property # this makes [self.loss_metric] the value of the self.metrics attribute
    def metrics(self):
        return [self.loss_metric]
    
    def train_step(self, batch):
        """Proccesses one batch inside model.fit()"""
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]

        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss = model.compute_loss(None, one_hot, preds, sample_weight=mask)

        trainable_vars = self.trainable_variables
        graidents = tape.gradients(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(graidents, trainable_vars))
        self.loss_metric.update_state(loss)

        return {"loss": self.loss_metric.result()}
    
    def test_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss = model.compute_loss(None, one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)

        return {"loss": self.loss_metric.result()}
    
    def generate(self, source, target_start_token_idx):
        """Preforms inference over one batch of inputs using greedy decoding"""
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32 * target_start_token_idx)
        dec_logits = []

        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input

keras.utils.get_file(
        origin="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        extract=True,
        archive_format="tar",
        cache_dir="."
        )

saveto = "./datasets/LJSpeech-1.1"
wavs = glob("{}/**/*.wav".format(saveto), recursive=True)

id_to_text = {}
with open(os.path.join(saveto, "metadata.csv"), encoding="utf-8") as f:
    for line in f:
        id = line.strip().split("|")[0]
        text = line.strip().split("|")[2]
        id_to_text[id] = text

def get_data(wavs, id_to_text, maxlen=50):
    """returns mapping of audio paths and transcription texts"""
    data = []
    for w in wavs:
        id = pattern_wav_name.split(w)[-4]
        if len(id_to_text[id]) < maxlen:
            data.append({"audio": w, "text": id_to_text[id]})
    
    return data