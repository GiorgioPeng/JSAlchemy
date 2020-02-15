import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { MnistData } from './data'
import { softmax, mod } from '@tensorflow/tfjs';

window.onload = async () => {
    const data = new MnistData();
    await data.load();//加载图片和二进制文件
    const example = data.nextTestBatch(50)//加载一些验证集/测试集
    // 图片是28px*20px的,还是黑白的,所以图片的维度是28*28*1 = 784, 如果是彩色图,就需要计算rgb三个通道的值,也就是 28*28*3 = 2352
    console.log(example)
    const surface = tfvis.visor().surface(
        { name: '输入示例' },
    )//做一个tfvis的侧边图
    for (let i = 0; i < 50; i++) {
        const imgTensor = tf.tidy(() => {
            return example.xs
                .slice([i, 0], [1, 784])//切割
                .reshape([28, 28, 1]);//转化成2维图像黑白(如果是彩色第三维就不是1)
        });
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = "margin:4"
        await tf.browser.toPixels(imgTensor, canvas);//渲染图像
        // document.body.appendChild(canvas)//添加到body中
        surface.drawArea.appendChild(canvas);//将图片添加到自己创建的tfvis的侧边图中
    }

    const model = tf.sequential()
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],//图片的高度,宽度和色彩(黑白为1,彩色为3)
        kernelSize: 5, //卷积核大小(就是卷积核这个函数对应需要乘的矩阵的维度,一般设置为奇数,有中心点)
        filters: 8,//卷积核的数量,超参数,可以调整
        strides: 1,//移动步长,卷积框移动的步数
        activation: 'relu',//  y = max(0,x) 当特征大于0的时候保留,小于0的时候舍弃掉,可以减少部分特征值
        kernelInitializer: 'varianceScaling'//卷积核初始化方法,可以加快收敛速度,也可以不设置
    }));//添加2维卷积层
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],//池化层大小
        strides: [2, 2]
    }));//添加最大2维池化层

    //进行组合,再来一轮特征提取
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        strides: 1,
        filters: 16,//需要多一点,因为是前面的排列组合
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],//池化层大小
        strides: [2, 2]
    }));//添加最大2维池化层

    //将高维图数据转化到一维
    model.add(tf.layers.flatten());

    //添加输出层
    model.add(tf.layers.dense({
        units: 10,//最终10个类别
        activation: 'softmax',
        kernelInitializer: 'varianceScaling'
    }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: 'accuracy'
    })

    const [inputs, labels] = tf.tidy(() => {
        const d = data.nextTestBatch(1500);
        return [
            d.xs.reshape([1500, 28, 28, 1]),
            d.labels
        ]
    })//训练集
    const [Yinputs, Ylabels] = tf.tidy(() => {
        const d = data.nextTestBatch(200);
        return [
            d.xs.reshape([200, 28, 28, 1]),
            d.labels
        ]
    })//验证集
    await model.fit(inputs,labels,{
        validationData:[Yinputs,Ylabels],
        epochs:50,
        callbacks:tfvis.show.fitCallbacks(
            {name:'损失'},
            ['loss','val_loss','acc','val_acc'],
            {callbacks:['onEpochEnd']}
        )
    })

    const canvas = document.querySelector('canvas')
    window.clear = () => {
        const context = canvas.getContext('2d');
        context.fillStyle = 'rgb(0,0,0)'
        context.fillRect(0, 0, 300, 300)
    }

    canvas.addEventListener('mousemove', (e) => {
        if (e.buttons === 1) {//如果是按着左键滑动的
            const context = canvas.getContext('2d');
            context.fillStyle = 'rgb(255,255,255)'
            context.fillRect(e.offsetX, e.offsetY, 10, 10)
        }
    })

    clear();

    window.predict = () => {
        const input = tf.tidy(() => {
            return tf.image.resizeBilinear(//重新定义大小
                tf.browser.fromPixels(canvas),
                [28,28],
                true
            ).slice([0,0,0],[28,28,1])//使用slice将颜色从彩色变成黑白(切掉两层高度)
            .toFloat().div(255)//归一化
            .reshape([1,28,28,1])//一个图片
        })//将canvas转化成tensor
        const pred = model.predict(input).argMax(1);
        alert(`预测结果为${pred.dataSync()[0]}`)
    }
}