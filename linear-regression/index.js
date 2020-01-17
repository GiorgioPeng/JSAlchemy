import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
window.onload = async () => {
  const xs = [1, 2, 3, 4]; //x轴训练数据
  const ys = [1, 3, 5, 7]; //y轴训练数据
  tfvis.render.scatterplot(
    { name: "样本" }, //图例的名称
    {
      values: xs.map((x, i) => {
        return { x: x, y: ys[i] };
      }) //数据为一个对象数组
    },
    {
      xAxisDomain: [0, 5],
      yAxisDomain: [0, 8]
    } //设定X,Y轴的范围
  );
  const model = tf.sequential(); //创建连续的模型,上一层的输出必然是下一层的输入,大多数都是这个模型

  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  //给模型添加一个全连接层,units该层神经元个数,inputShape输入数据的维度
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.1)
  }); //添加 均方误差 损失函数, 优化器(和学习速率)

  //将数据转化成tensor
  const inputs = tf.tensor(xs); //输入
  const labels = tf.tensor(ys); //给定的输出
  await model.fit(inputs, labels, {
    batchSize: 4, //每次模型要学的样本的数据量大小
    epochs: 200, //迭代学习次数
    callbacks: tfvis.show.fitCallbacks({ name: "训练过程" }, ["loss"])
  });
};
