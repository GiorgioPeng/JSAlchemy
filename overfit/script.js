import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

// //欠拟合
// import { getData } from '../xor/data'
// window.onload = async () => {
//     const data = getData(200);
//     tfvis.render.scatterplot(
//         { name: '欠拟合' },
//         {
//             values: [data.filter((p) => p.label === 1), data.filter((p) => p.label === 0)]
//         }
//     )
//     const model = tf.sequential();
//     model.add(tf.layers.dense({
//         units:1,
//         inputShape:[2],
//         activation:'sigmoid'
//     }))
//     model.compile({
//         loss:tf.losses.logLoss,
//         optimizer:tf.train.adam(0.1)
//     })
//     const inputs = tf.tensor(data.map(p=>[p.x,p.y]))
//     const labels = tf.tensor(data.map(p=>p.label))
//     await model.fit(inputs,labels,{
//         validationSplit:0.2,//设置部分验证集
//         epochs:200,
//         callbacks:tfvis.show.fitCallbacks(
//             {name:'损失'},
//             ['loss','val_loss'],
//             {callbacks:['onEpochEnd']}
//         )
//     })
// }

//过拟合
import { getData } from './data'

window.onload = async () => {
    const data = getData(200, 2);
    tfvis.render.scatterplot(
        { name: '过拟合' },
        { values: [data.filter(p => p.label === 1), data.filter(p => p.label === 0)] }
    )
    const model = tf.sequential();
    model.add(tf.layers.dense(
        {
            units: 10,
            inputShape: [2],
            activation: 'tanh'
        }
    ))
    model.add(tf.layers.dense(
        {
            units: 10,
            activation: 'tanh',
            // kernelRegularizer:tf.regularizers.l2({l2:1})//设置权重衰减,一般在最复杂的一层设置,l2方法内设置的是正则化率(超参数)
        }
    ))
    //丢弃法,在最复杂的隐藏层后加一层,参数为丢弃率,假如有10个神经元,丢弃率设置为0.9,就是随机丢弃掉9个神经元
    model.add(tf.layers.dropout({rate:0.5}))

    model.add(tf.layers.dense(
        {
            units: 1,
            activation: 'sigmoid'
        }
    ))
    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
    });

    const inputs = tf.tensor(data.map(p => [p.x, p.y]))
    const labels = tf.tensor(data.map(p => p.label))

    await model.fit(inputs, labels, {
        epochs: 200,
        validationSplit: 0.2,
        callbacks: tfvis.show.fitCallbacks(
            { name: '损失' },
            ['loss', 'val_loss'],
            { callbacks: ['onEpochEnd'] }
        )
        
    })
}
