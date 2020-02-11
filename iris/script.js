import { getIrisData, IRIS_CLASSES } from './data'

window.onload = () => {
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15); //%15的数据作为验证集,返回的4个都是tensor, xTrain训练集的元数据，yTrain训练集的结果,xTest验证集的元数据，yTest验证集的结果 
    xTrain.print();
    yTrain.print();
    xTest.print();
    yTest.print();
    console.log(IRIS_CLASSES)
}