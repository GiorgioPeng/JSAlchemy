const loadImg = src => {
    return new Promise((resolve, reject) => {
        let img = new Image();
        img.crossOrigin = 'anonymous'//使得图片src能够跨域
        img.src = src;
        img.height = 224;
        img.width = 224;
        img.onload = () => {
            resolve(img)
        }
    })
}
export async function getImg() {
    const imgPromiseArray = [];
    const data = []
    const labels = []
    for (let i = 0; i < 30; i++) {
        data.push(`android-${i}.jpg`),labels.push([1,0,0])
        data.push(`apple-${i}.jpg`),labels.push([0,2,0])
        data.push(`windows-${i}.jpg`),labels.push([0,0,1])
    }
    for (let i of data) {
        imgPromiseArray.push(loadImg('http://127.0.0.1:8080/brand/train/' + i))//将所有图片的promise放在一个数组中
    }
    const inputs = await Promise.all(imgPromiseArray)
    return {inputs,labels}
}