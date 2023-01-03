import tensorflow as tf


def psnr(y_true, y_pred):
    y_true = tf.cast(y_true*255, tf.uint8)
    y_pred = tf.cast(y_pred*255, tf.uint8)
    return tf.image.psnr(y_true, y_pred, max_val=255)


def ssim(y_true, y_pred):
    y_true = tf.cast(y_true*255, tf.uint8)
    y_pred = tf.cast(y_pred*255, tf.uint8)
    return tf.image.ssim(y_true, y_pred, max_val=255)
