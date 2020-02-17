export function file2img(file) {
    return new Promise(resolve=>{
        const reader = new FileReader();
        reader.readAsDataURL(file);//读取文件,是一个异步操作,回调函数需要写在onload中(下面)
        reader.onload = (e) => {
            const img = new Image();//也是一个异步操作,完成的回调函数也应该写在onload中
            img.src = e.target.result;
            img.width = 224//这里是因为预训练模型,接受图片宽度为224
            img.height = 224//同上
            img.onload = ()=>{
                resolve(img)
            }
        }})
}