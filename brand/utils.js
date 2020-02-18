import * as tf from '@tensorflow/tfjs'
export function img2x(el) {
    return tf.tidy(() => tf.browser.fromPixels(el)//将图片转化成tensor
        .toFloat()
        .div(177.5)
        .sub(1)//归一化
        .reshape([1, 224, 224, 3])
    )
}