# Implementation of Dawid-Skene Model with Edward
#
# i = 1, 2, ..., N : number of workers
# j = 1, 2, ..., J : number of tasks
#
# p({x_ij}, {t_i}) = \prod_{i, j} p(x_i | t_i) p(t_i)
# 
# x^_ij : response of worker j for task i (x \in {0, 1})
# t_i : ground truth of task i (t \in {0, 1})
#
# p(x_ij = 1 | t_i = 0) = \alpha^{j}_{0}
# p(x_ij = 1 | t_i = 1) = \alpha^{j}_{1}

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import edward as ed

from edward.models import Dirichlet, Multinomial, Beta, Bernoulli, PointMass

from sklearn.metrics import roc_auc_score


# data
N = 50
K = 10

t_true = np.random.randint(0, 2, size=[N])
t_true_2D = np.array([t_true, 1-t_true])
alpha_true = np.random.beta(a=1, b=1, size=[K, 2])

x_data = np.random.rand(K, N) < np.dot(alpha_true, t_true_2D)
x_data = x_data + 0

# model
pi = Dirichlet(concentration=tf.ones(2))
t = Multinomial(total_count=1., probs=pi, sample_shape=N)
alpha = Beta(concentration0=tf.ones([K, 2]), concentration1=tf.ones([K, 2]))
X = Bernoulli(probs=tf.matmul(alpha, tf.transpose(t)))

# inference
qpi = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([2]))))
qt = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([N, 2]))))
qalpha = PointMass(params=tf.nn.sigmoid(tf.Variable(tf.random_normal([K, 2]))))

inference = ed.MAP({pi: qpi, t: qt, alpha: qalpha}, data={X: x_data})

inference.run(n_iter=5000)


# criticism
t_pred = qt.mean().eval().argmax(axis=1)
accuracy = (N - np.count_nonzero(t_pred - t_true)) / N
t_prob = qt.mean().eval()[:, 1]
auc = roc_auc_score(t_true, t_prob)

## label flip may occur
if auc < 0.5:
    t_pred = 1 - t_pred
    accuracy = 1. - accuracy
    auc = 1. - auc

print('t_pred')
print(t_pred)
print('t_true')
print(t_true)


print('accuracy')
print(accuracy)


print('AUC')
print(auc)
