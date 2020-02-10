export function getData(numSamples) {
    //numSamples 生成点的数量（蓝色点和黄色点加起来） 
    let points = [];
    function genGauss(cx, cy, label) {
        for (let i = 0; i < numSamples / 4; i++) {
            let x = normalRandom(cx);//x 以 cx 为中心的正太分布
            let y = normalRandom(cy);//y 以 cy 为中心的正太分布
            points.push({ x, y, label }); //两者组合或者一个二维正太分布
        }
    }
    genGauss(2, 2, 1);
    genGauss(-2, -2, 1);
    genGauss(-2, 2, 0);
    genGauss(2, -2, 0);
    return points;
}
function normalRandom(mean = 0, variance = 1) {
    let v1, v2, s;
    do {
        v1 = 2 * Math.random() - 1;
        v2 = 2 * Math.random() - 1;
        s = v1 * v1 + v2 * v2;
    } while (s > 1); //s<1来确定s在圆圈内
    let result = Math.sqrt((-2 * Math.log(s)) / s) * v1;
    return mean + Math.sqrt(variance) * result;
}