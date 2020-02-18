import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getImg } from './data'
import { img2x } from './utils';
import { model } from '@tensorflow/tfjs';

const MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json'
const NUMBER_OF_OUTPUT = 3;//分类数量
const CLASS_NAME = ['android', 'apple', 'windows']

window.onload = async () => {
    const { inputs, labels } = await getImg();
    const surface = tfvis.visor().surface(
        { name: '输入示例', styles: { heigh: 250 } }
    )
    inputs.forEach(el => {
        surface.drawArea.appendChild(el)
    })
    const model1 = await tf.loadLayersModel(MODEL_PATH)
    model1.summary();//查看模型概况
    const layer = model1.getLayer('conv_pw_13_relu')//通过层的姓名获取中间层
    const truncatedMobilenet = tf.model({
        inputs: model1.inputs,
        outputs: layer.output,
    })//定义截断模型,将原始模型的输入作为现在模型的输入,将指定层的输出作为现在模型的输出

    const model2 = tf.sequential()
    model2.add(tf.layers.flatten({
        inputShape: layer.outputShape.slice(1),
        //将截断的输出形状拿出来作为这个模型的输入(需要slice是因为数组的第一个是输出的个数,而这个个数一般是不确定的,是一个null值,所以需要切割掉)
    }))//通过flatten层将高维特征摊平成一维向量以便于分类
    model2.add(tf.layers.dense({
        units: 10,
        activation: 'relu'
    }))
    model2.add(tf.layers.dense({
        units: NUMBER_OF_OUTPUT,
        activation: 'softmax'
    }))
    model2.compile({ optimizer: tf.train.adam(), loss: 'categoricalCrossentropy' })
    const { xs, ys } = tf.tidy(() => {
        const xs = tf.concat(inputs.map(img => truncatedMobilenet.predict(img2x(img))))
        //直接使用截断模型进行预测,获得能够作为新模型输入的结果,然后使用tf.concat方法,将这些tensor加一维,变成一个高一维的tensor
        const ys = tf.tensor(labels)
        return { xs, ys }

    })
    await model2.fit(xs, ys, {
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss'],
            { callbacks: ['onEpochEnd'] }
        )
    })//训练新模型

    window.predict = file => {
        let fileReader = new FileReader()
        fileReader.readAsDataURL(file);
        fileReader.onload = e => {
            const img = new Image()
            img.src = e.target.result;
            img.height = 224;
            img.width = 224;
            img.onload = () => {
                document.body.appendChild(img)
                const input = tf.tidy(() => img2x(img))
                const results = model2.predict(truncatedMobilenet.predict(input));
                setTimeout(() => { alert(CLASS_NAME[results.argMax(1).dataSync()[0]]) }, 0)
            }
        }
    }

    window.download = async ()=>{
        await model2.save('downloads://model')//模型保存
    }
}