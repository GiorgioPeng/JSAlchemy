import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { MnistData } from './data'

window.onload = async () => {
    const data = new MnistData();
    await data.load();//加载图片和二进制文件
    const example = data.nextTestBatch(20)//加载一些验证集/测试集
    for(let i = 0 ;i<20;i++){
        
    }
}