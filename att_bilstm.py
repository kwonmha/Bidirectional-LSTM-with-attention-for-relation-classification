# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def get_length(sequence):
		used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
		length = tf.reduce_sum(used, 1)
		length = tf.cast(length, tf.int32)
		return length


def get_by_label(tensor, label, n_classes):
	labels = tf.argmax(label, 1, output_type=tf.int32)  # (?, )
	label_index = tf.range(0, tf.shape(tensor)[0]) * n_classes + labels
	flat_tensor = tf.reshape(tensor, [-1])
	return tf.gather(flat_tensor, label_index), label_index  # (?, 19) -> (?,)


def get_neg_out(tensor, label, n_classes):
	print("input tensor shape : ", np.shape(tensor))
	labels = tf.argmax(label, 1, output_type=tf.int32)  # (?, )
	# label_index = tf.range(0, tf.shape(tensor)[0]) * n_classes + labels

	top2v, top2i = tf.nn.top_k(tensor, 2)
	# max_index = top2i[:, 0]
	max_index = tf.range(0, tf.shape(tensor)[0]) * n_classes + top2i[:, 0]

	lab_max_comp = tf.equal(labels, top2i[:, 0])  # 라벨이랑 같으면 true true.. 다르면 false => false를 찾고
	# lab_2max_comp = tf.equal(label_index, top2i[1])  #true인 애들을 찾고
	index = tf.where(lab_max_comp, x=top2i[:, 1], y=top2i[:, 0])
	neg_index = tf.range(0, tf.shape(tensor)[0]) * n_classes + index
	flat_tensor = tf.reshape(tensor, [-1])
	return tf.gather(flat_tensor, neg_index), neg_index, max_index, lab_max_comp, top2i[:, 0]


class BiLSTMAttention(object):
	def __init__(self, layers, max_length, n_classes, vocab_size, embedding_size=300, batch_size = 64, gamma=2.0, mp=2.5, mn=0.5, use_ranking_loss=False):
		self.input_text = tf.placeholder(tf.int32, shape=[None, max_length], name="input_text")
		self.labels = tf.placeholder(tf.int32, shape=[None, n_classes])
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

		self.W_emb = tf.get_variable("word_embeddings", [vocab_size, embedding_size])
		rnn_input = tf.nn.embedding_lookup(self.W_emb, self.input_text)

		seq_len = get_length(rnn_input)

		layers = list(map(int, layers.split('-')))
		cells = [tf.nn.rnn_cell.LSTMCell(h, activation=tf.nn.tanh, state_is_tuple=True) for _, h in enumerate(layers)]
		multi_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
		rnn_out, _state = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_cells, cell_bw=multi_cells, inputs=rnn_input, sequence_length=seq_len, dtype=tf.float32)
		rnn_out = tf.concat(rnn_out, 2)

		att_vec_param = tf.get_variable("att_vec_param", [layers[-1] * 2])
		tan_rnn_out = tf.tanh(rnn_out)
		# att_vec = tf.tensordot(sim_mat_tan, tf.transpose(att_vec_param), axes=1) # (B,T,D) * (D, 1)=> (B, T)
		att_vec = tf.einsum('aij,j->ai', tan_rnn_out, tf.transpose(att_vec_param))
		att_sm_vec = tf.nn.softmax(att_vec)
		rnn_attention = tf.einsum('aij,ai->aj', rnn_out, att_sm_vec)  # (B, T, D) * (B, T) => (B, D)

		max_val = np.power(6/(n_classes + layers[-1]), 1/2)
		C_emb = tf.get_variable("c_emb", [layers[-1] * 2, n_classes], initializer=tf.random_uniform_initializer(minval=-max_val, maxval=max_val))
		output = tf.matmul(rnn_attention, C_emb)  # (B, D) * (D, C) => (B, C)
		self.output = output

		if use_ranking_loss:
			pos_out, pos_index = get_by_label(output, self.labels, n_classes)
			self.pos_out = pos_out
			self.pos_index = pos_index

			# neg_out, neg_index, max_index, cond, top2i = get_neg_out(output, self.labels, n_classes)
			# self.neg_out = neg_out
			# self.neg_index = neg_index
			# self.max_index = max_index
			# self.cond = cond
			# self.top2i = top2i

			# self.cost = tf.reduce_sum(tf.log(1 + tf.exp(gamma * (mp - pos_out))) + tf.log(1 + tf.exp(gamma * (mn - neg_out))))
			self.cost = tf.reduce_sum(tf.log(1 + tf.exp(gamma * (mp - pos_out))))
		else:
			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.labels))

		optimizer = tf.train.AdamOptimizer()
		self.train = optimizer.minimize(self.cost)

		self.predictions = tf.argmax(output, 1, name="predictions")
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 1)), tf.float32))


