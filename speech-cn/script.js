import * as speechCommand from '@tensorflow-models/speech-commands'
import * as tfvis from '@tensorflow/tfjs-vis'
import { callbacks } from '@tensorflow/tfjs';

const PATH = 'http://127.0.0.1:8080/slider/'
let transferRec;
window.onload = async () => {
    const rec = speechCommand.create('BROWSER_FFT')//创建识别器   
    await rec.ensureModelLoaded()//确保加载完成
    transferRec = rec.createTransfer('轮播图')//创建迁移学习器

    //设置收集方法,涉及io操作,一般都是异步
    window.collect = async (btn) => {
        btn.disabled = true;
        const label = btn.innerHTML;
        await transferRec.collectExample(
            label === '背景噪音' ? '_background_noise_' : label
        )//调用浏览器接口录音,点背景噪音点时候不需要发出声音
        btn.disabled = false;
        document.querySelector('pre').innerHTML = JSON.stringify(transferRec.countExamples(), null, 2);
    }

    //模型训练
    window.train = async () => {
        await transferRec.train(
            {
                epochs: 30,
                callback: tfvis.show.fitCallbacks(
                    { name: '训练效果' },
                    ['loss', 'acc'],
                    { callbacks: ['onEpochEnd'] }
                )
            }
        )
    }

    //开关摄像头
    window.toggle = async (checked) => {
        if (checked === true) {
            transferRec.listen(result => {
                const {scores} = result;
                const labels = transferRec.wordLabels();
                const index= scores.indexOf(Math.max(...scores));
                console.log(labels[index])
            },
                {
                    overlapFactor: 0,
                    probabilityTHreshold: 0.7
                })
        }
        else{
            transferRec.stopListening( )
        }
    }

    //数据下载
    window.save = ()=>{
        const arrayBuffer = transferRec.serializeExamples();//对训练对数据进行序列化,返回一个arraybuffer
        const blob = new Blob([arrayBuffer])//将arraybuffer转化成blob方便下载

        //下载操作
        const link = document.createElement('a')
        link.href = window.URL.createObjectURL(blob)
        link.download='data.bin'
        link.click();

    }
}