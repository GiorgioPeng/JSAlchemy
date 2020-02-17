import * as tf from '@tensorflow/tfjs'
import { file2img } from './util'
import { IMAGENET_CLASSES } from './imagenet_classes'
//导入预训练模型地址
const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json'

window.onload = async () => {
    //加载预训练模型
    const model = await tf.loadLayersModel(MOBILENET_MODEL_PATH)

    window.predict = async (file) => {
        const img = await file2img(file)
        document.body.appendChild(img)
        const pred = tf.tidy(() => {
            const input = tf.browser.fromPixels(img)//将图片转化成tensor
                .toFloat()
                .div(177.5)
                .sub(1)//归一化
                .reshape([1, 224, 224, 3])
            return model.predict(input)
        });
        setTimeout(() => {
            alert(IMAGENET_CLASSES[pred.argMax(1).dataSync()[0]])
        }, 0);//将弹出扔出任务队列,防止alert事件阻塞图片加载
    }
}