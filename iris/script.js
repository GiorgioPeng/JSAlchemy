import { getIrisData, IRIS_CLASSES } from './data'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

window.onload = async () => {
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15); //%15的数据作为验证集,返回的4个都是tensor, xTrain训练集的元数据，yTrain训练集的结果,xTest验证集的元数据，yTest验证集的结果 
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 10,//随意选取的神经元
        inputShape: [xTrain.shape[1]],//获取输入数据的shape
        activation: "sigmoid"//只要带来非线性变化的都可以
    }))

    //多分类神经网络的核心代码
    model.add(tf.layers.dense({
        units: 3,//第二层的神经元个数必须是输出类别的个数，吧燕尾花分为3类
        activation: 'softmax'//3个概率（和为1），也就对应分成3类

    }))
    //softmax 交叉熵损失函数：对数损失函数对多分类版本（对数损失函数是交叉熵损失函数的一种特殊形式，即对数损失函数用于处理2分类数据）

    model.compile({
        loss: 'categoricalCrossentropy', 
        optimizer: tf.train.adam(0.1),
        metrics: ['accuracy']//设置准确度
    })
    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest],//设置验证集
        callbacks: tfvis.show.fitCallbacks(
            {name:'训练效果'},
            ['loss','val_loss','acc','val_acc'],//训练集损失，验证集损失，训练集准确度，验证集准确度
            //注意，这些图例度名称不是任意的，必须是这几个，才能正确显示
            //如果训练集和验证集的曲线相差不大，说明超参数配置合理
            {callbacks:[ 'onEpochEnd']}
        )

    })

    window.predict = (form)=>{
        const input = tf.tensor([[
            form.a.value*1,
            form.b.value*1,
            form.c.value*1,
            form.d.value*1
        ]])
        const pred = model.predict(input)
        console.log(pred.dataSync(0))
        alert(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`)
        //.argMax 方法可以输出某个维度的最大值的索引 0=> 第一维， 1=>第二维
    }
}