import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";
import { getData } from "./creatData.js";
import { callbacks } from "@tensorflow/tfjs";
window.onload = async () => {
  const data = getData(400);
  // console.log(data);
  tfvis.render.scatterplot(
    {
      name: "逻辑回归"
    },
    // { values: data.map(d => ({ x: d.x, y: d.y })) }
    {
      values: [data.filter(d => d.label === 1), data.filter(d => d.label === 0)]
    } //values可以为一维数组或者二维数组,当values为一维数组的时候,会直接做图
    //当values为二维数组的时候, 可以将图的颜色分开
    // {
    //   xAxisDomain: [-4, 4],
    //   yAxisDomain: [-4, 4]
    // }
  );
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 1, inputShape: [2], activation: "sigmoid" })
  );
  //这次试用SIGMOD激活函数是为了将输出压缩到0-1之间,使得其像概率
  //inputShape为2是因为相对于之前的线性回归(输入仅为X值,),逻辑回归需要同时判断X,Y的值来做出决定(tf.tensor([1,2])类似这种)
  //units为1因为只需要一层就可以得到结果

  model.compile({ loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1) });
  const inputs = tf.tensor(data.map(p => [p.x, p.y]));
  const labels = tf.tensor(data.map(p => p.label));
  console.log(inputs);
  await model.fit(inputs, labels, {
    batchSize: 40, //十组batch就完成了一组epoch
    epochs: 20,
    callbacks: tfvis.show.fitCallbacks(
      {
        name: "训练过程"
      },
      ["loss"]
    )
  });
  window.predict = form => {
    const pred = model.predict(
      tf.tensor([[form.x.value * 1, form.y.value * 1]])
    );
    alert(`预测结果${pred.dataSync()[0]}`);
  };
};
