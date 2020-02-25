import * as speechCommands from '@tensorflow-models/speech-commands'

const PATH = 'http://127.0.0.1:8080/slider/data.bin'

let transferRec;
window.onload = async () => {
    const rec = speechCommands.create('BROWSER_FFT')
    await rec.ensureModelLoaded()
    transferRec = rec.createTransfer('轮播图')
    const res = await fetch(PATH)//下载数据

    const arrayBuffer = await res.arrayBuffer()//将数据转化成arrayBuffer

    transferRec.loadExamples(arrayBuffer)//给迁移学习期加载数据

    console.log(transferRec.countExamples())//查看数据

    await transferRec.train({
        epochs: 30
    })//训练数据

    console.log('训练完成')

    window.toggle = async  (checked) => {
        if (checked) {
            await transferRec.listen((result) => {
                const { scores } = result;//或者结果分数
                const label = transferRec.wordLabels();//获得可识别列表
                const index = scores.indexOf(Math.max(...scores))//获得最大值索引
                console.log(label[index])
                window.play(label[index]);
            }, {
                overlapFactor: 0.01,
                probabilityTHreshold: 0.65
            })
        }
        else {
            transferRec.stopListening();
        }
    }
    let currentIndex = 0;
    window.play = label =>{
        const div = document.querySelector(".slider>div");
        if(label === '上一张'&&currentIndex){
            currentIndex--;
        }
        else{
            if(label === '下一张'&&currentIndex!==(document.querySelectorAll('img').length-1))
                currentIndex++;
        }
        div.style.transition = `transform 1s linear`
        div.style.transform = `translateX(-${100*currentIndex}%)`
    }
}