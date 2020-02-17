
import * as speechcommand from '@tensorflow-models/speech-commands'

const MODEL_PATH = 'http://127.0.0.1:8080/speech'
window.onload = async () => {
    const rec = speechcommand.create(
        'BROWSER_FFT',//浏览器自带的傅立叶变换
        null,
        MODEL_PATH + '/model.json',
        MODEL_PATH + '/metadata.json'
    )//创建识别器

    await rec.ensureModelLoaded();//确保模型加载完成
    
    let labels = rec.wordLabels()//获得可识别的字符
    labels = labels.slice(2)
    const div  = document.querySelector("#result")
    for (let i of labels.map( l => `<div style="width:30%;height:100px;margin-top:5px;text-align:center;line-height:100px;font-size:20px;border:2px solid #999">${l}</div>`))
        div.innerHTML+=i;
    

 
    rec.listen(result => {
        const { scores } = result
        let index = scores.indexOf(Math.max(...scores))
        console.log(labels[index])
        div.innerHTML = '';
        for (let i of labels.map( (l,i) => `<div style="background-color:${i+2===index&&'green'};width:30%;height:100px;margin-top:5px;text-align:center;line-height:100px;font-size:20px;border:2px solid #999">${l}</div>`))
            div.innerHTML+=i;
    
    },
        {
            overlapFactor:0.2,//覆盖率
            probabilityThreshold: 0.75//相似度
        })
}