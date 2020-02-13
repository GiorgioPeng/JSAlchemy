import * as tfvis from '@tensorflow/tfjs-vis'
import * as tfjs from '@tensorflow/tfjs'
import { getData } from './data'
window.onload = async () => {
    const points = getData(400);
    tfvis.render.scatterplot(
        { name: 'xor' },
        {
            values: [points.filter(point => point.label === 1), points.filter(point => point.label === 0)]
        }
    )
    const model = tfjs.sequential();
    model.add(tfjs.layers.dense({
        units: 4,
        inputShape: [2],
        activation: 'relu'
    })); //设置隐藏层，只有第一层需要设置inputshape，如果把隐藏层中的激活函数去掉，则这个隐藏层没有任何作用（线性组合还是线性，无法解决非线性问题）
    model.add(tfjs.layers.dense({
        units: 1, activation: 'sigmoid'
    }));//设置输出层
    model.compile({ loss: tfjs.losses.logLoss, optimizer: tfjs.train.adam(0.1) });//补充损失函数和优化器,logloss常用于逻辑损失函数
    const inputs = tfjs.tensor(points.map(point => [point.x, point.y]));
    const labels = tfjs.tensor(points.map(point => point.label));
    await model.fit(inputs, labels, {
        batchSize: 40,
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
            {
                name: '训练过程'
            },
            ['loss']
        )
    })
    window.predict = async (form) => {
        const pred = await model.predict(
            tfjs.tensor(
                [
                    [
                        form.x.value * 1, form.y.value * 1//乘1将字符串转化乘数字
                    ]
                ]
            )
        )
        alert(`预测结果：${pred.dataSync()[0]}`)
    }
}