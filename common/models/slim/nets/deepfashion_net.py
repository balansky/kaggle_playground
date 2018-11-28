from tensorflow.contrib import slim
from tensorflow.contrib import layers
import tensorflow as tf
# from . import vgg

def landmark_pool(conv4, landmark_vis, landmark_loc, num_landmarks):
    batch_size = tf.shape(conv4)[0]
    def crop_landmark(idx):
        land_res = []
        l_vis = landmark_vis[idx]
        l_loc = landmark_loc[idx]
        conv5_pick = tf.expand_dims(conv4[idx], axis=0)
        def landmark_pool(landmark_height, landmark_width):
            res_img = tf.image.extract_glimpse(conv5_pick, [5, 5], [[landmark_height, landmark_width]],
                                               centered=False, normalized=True)
            pool_res = slim.max_pool2d(res_img,  [2, 2], stride=1)
            pool_res = tf.squeeze(pool_res, 0)
            # pool_res = tf.reshape(pool_res, [4, 512])
            return pool_res

        eight_l_vis = tf.unstack(l_vis, axis=0)
        eight_l_loc = tf.unstack(l_loc, axis=0)
        for j in range(num_landmarks):
            p = landmark_pool(eight_l_loc[j][0], eight_l_loc[j][1])
            lm_pool = tf.cond(tf.greater(eight_l_vis[j], 0.5), lambda: p, lambda: tf.zeros([4, 4, 512]))
            land_res.append(lm_pool)
        return tf.concat(land_res, axis=2)

    indices = tf.range(batch_size)
    res = tf.map_fn(crop_landmark, indices, tf.float32, name='landmark_pool')
    return res


def block_landmark(conv4):
    with tf.variable_scope("landmark"):
        conv4_pool = slim.max_pool2d(conv4, [2, 2], scope='pool4')
        conv5 = slim.repeat(conv4_pool, 3, layers.conv2d, 512, [3, 3], scope='conv5')
        conv5_pool = slim.max_pool2d(conv5, [2, 2], scope='pool5')
        conv6_pose = slim.conv2d(conv5_pool, 1024, [7, 7], padding='VALID', scope='fc6_pose')
    return conv6_pose



def block_fushion(conv4, landmark_vis, landmark_loc, num_landmarks):
    with tf.variable_scope("global"):
        conv4_pool = slim.max_pool2d(conv4, [2, 2], scope='pool4')
        conv5 = slim.repeat(conv4_pool, 3, layers.conv2d, 512, [3, 3], scope='conv5')
        conv5_pool = slim.max_pool2d(conv5, [2, 2], scope='pool5')
        fc6_global = slim.conv2d(conv5_pool, 4096, [7, 7], padding='VALID', scope='fc6')
    with tf.variable_scope("local"):
        res = landmark_pool(conv4, landmark_vis, landmark_loc, num_landmarks)
        fc6_local = slim.conv2d(res, 1024, [4, 4], padding='VALID', scope='fc6')
    with tf.variable_scope("fusion"):
        fc_concat = tf.concat([fc6_global, fc6_local], axis=3, name='fc_concat')

    return fc_concat


def deepfashion_vgg_landmark(conv4, num_landmarks, is_training, dropout_keep_prob=1.0):

    with tf.variable_scope('landmark_block'):
        net = block_landmark(conv4)
        net = slim.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout6_pose')
        net = slim.conv2d(net, 1024, [1, 1], scope='fc7_pose')
        net = slim.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout7_pose')
        landmark_v_logits = slim.conv2d(net, num_landmarks, [1, 1], activation_fn=None,
                                        normalizer_fn=None, scope='visible_logits')
        landmark_v = tf.squeeze(landmark_v_logits, [1, 2], name='visible_logits/squeezed')

        landmark_loc_logits = slim.conv2d(net, num_landmarks * 2, [1, 1], activation_fn=None,
                                          normalizer_fn=None, scope='loc_logits')

        landmark_loc = tf.squeeze(landmark_loc_logits, [1, 2], name='loc_logits/squeezed')
    return landmark_v, landmark_loc, net



def deepfashion_vgg_fusion(conv4, landmark_v_sigmoid, landmark_loc, num_landmarks, is_training, dropout_keep_prob=1.0):
    with tf.variable_scope('fushion_block'):
        net = block_fushion(conv4, landmark_v_sigmoid, landmark_loc, num_landmarks)
        concat_dropout = slim.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout_concat')

        fc_fusion = slim.conv2d(concat_dropout, 4096, [1, 1], scope='landmark_loc')

        fusion_dropout = slim.dropout(
            fc_fusion, dropout_keep_prob, is_training=is_training, scope='dropout_fusion')
    return fusion_dropout, net



# def deepfashion_vgg_inference(inputs, num_classes, num_landmarks, is_training, dropout_keep_prob=1.0,
#                               scope='DeepfashionVgg'):
#     with tf.variable_scope(scope):
#         with slim.arg_scope(vgg.vgg_arg_scope()):
#             net, end_points = vgg.vgg_16(inputs, num_classes, is_training=is_training,
#                                          dropout_keep_prob=dropout_keep_prob)
#             conv4 = end_points['vgg_16/conv4/conv4_3']
#             landmark_v_logits, landmark_loc_logits, net = deepfashion_vgg_landmark(conv4, num_landmarks, is_training,
#                                                                                    dropout_keep_prob)
#             landmark_loc_logits = tf.reshape(landmark_loc_logits, [-1, num_landmarks, 2])
#             end_points['landmark_block'] = net
#             landmark_v_sigmoid = tf.nn.sigmoid(landmark_v_logits)
#             fusion_dropout, net = deepfashion_vgg_fusion(conv4, landmark_v_sigmoid, landmark_loc_logits, num_landmarks,
#                                                          is_training, dropout_keep_prob)
#
#             category_logits = slim.conv2d(fusion_dropout, num_classes, [1, 1], activation_fn=None,
#                                           normalizer_fn=None, scope='category/logits')
#
#             category_logits_squeezed = tf.squeeze(category_logits, [1, 2], name='category/logits/squeezed')
#
#     return category_logits_squeezed, landmark_v_logits, landmark_loc_logits, net
#
#
#
# def deepfashion_vgg(inputs, landmark_vis, landmark_loc, num_classes, num_landmarks, is_training, dropout_keep_prob=1.0,
#                     scope='DeepfashionVgg'):
#     with tf.variable_scope(scope):
#         with slim.arg_scope(vgg.vgg_arg_scope()):
#             net, end_points = vgg.vgg_16(inputs, num_classes, is_training=is_training,
#                                          dropout_keep_prob=dropout_keep_prob)
#             conv4 = end_points[scope + '/vgg_16/conv4/conv4_3']
#             landmark_v_logits, landmark_loc_logits, net = deepfashion_vgg_landmark(conv4, num_landmarks, is_training,
#                                                                                    dropout_keep_prob)
#             landmark_loc = tf.reshape(landmark_loc, [-1, num_landmarks, 2])
#             fusion_dropout, net = deepfashion_vgg_fusion(conv4, landmark_vis, landmark_loc, num_landmarks, is_training,
#                                                          dropout_keep_prob)
#
#             category_logits = slim.conv2d(fusion_dropout, num_classes, [1, 1], activation_fn=None,
#                                           normalizer_fn=None, scope='category/logits')
#
#             category_logits_squeezed = tf.squeeze(category_logits, [1, 2], name='category/logits/squeezed')
#     return category_logits_squeezed, landmark_v_logits, landmark_loc_logits, net


def deepfashion_vgg_variables(scope='DeepfashionVgg'):
    trainable_vars = []
    excluded_vars = ['vgg_16/conv5/conv5_1/weights:0', 'vgg_16/conv5/conv5_1/biases:0',
                     'vgg_16/conv5/conv5_2/weights:0', 'vgg_16/conv5/conv5_2/biases:0',
                     'vgg_16/conv5/conv5_3/weights:0', 'vgg_16/conv5/conv5_3/biases:0',
                     'vgg_16/fc6/weights:0', 'vgg_16/fc6/biases:0', 'vgg_16/fc7/weights:0',
                     'vgg_16/fc7/biases:0',
                     'vgg_16/fc8/weights:0', 'vgg_16/fc8/biases:0']
    for var in tf.trainable_variables():
        if scope +'/' + var.name not in excluded_vars:
            trainable_vars.append(var)
    return trainable_vars