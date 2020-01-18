import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";
window.onload = async () => {
  const height = [159, 160, 170];
  const weight = [50, 60, 80];
  tfvis.render.scatterplot(
    { name: "身高体重" },
    {
      values: height.map((x, i) => {
        return { x: x, y: weight[i] };
      })
    },
    {
      xAxisDomain: [130, 180],
      yAxisDomain: [40, 90]
    }
  );
  //归一化操作
  const inputs = tf
    .tensor(height) //先将原始数据转化成tensor,
    .sub(159) //然后每个都减去原始数据中最小的数据
    .div(11); //最后每个都除以最大值和最小值的差
  inputs.print();
  const labels = tf
    .tensor(weight)
    .sub(50)
    .div(30);
  labels.print();
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.3)
  });
  await model.fit(inputs, labels, {
    batchSize: 3, //每次模型要学的样本的数据量大小
    epochs: 100, //迭代学习次数
    callbacks: tfvis.show.fitCallbacks({ name: "训练过程" }, ["loss"])
  });
  const output = model.predict(
    tf
      .tensor([180])
      .sub(159)
      .div(11)
  ); //预测数据也需要先归一化
  output
    .mul(30)
    .add(50)
    .print();
  console.log(output.dataSync()[0] * 30 + 50);
};
