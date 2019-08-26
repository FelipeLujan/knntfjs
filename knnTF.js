const tf = require("@tensorflow/tfjs");
const coordinates = tf.tensor([
  [-121, 47],
  [-121.2, 46.5],
  [-122, 46.4],
  [-120.9, 46.7]
]);

const prices = tf.tensor([[200], [250], [215], [240]]);
const predictionPoint = tf.tensor([-121, 47]);
const  k = 2
// let calculations = coordinates
//   .sub(predictionPoint) // calculating distance from each point to the prediction point
//   .pow(2) // calculating distance from each point to the prediction point
//   .sum(1) // calculating distance from each point to the prediction point
//   .pow(0.5) // calculating distance from each point to the prediction point
//   .expandDims(1) // transform the distances to a vertical 1d tensor
//   .concat(prices, 1) // concat the prices to the distances, making it a 2d tensor
//   .unstack() // the [4, 2] tensor becomes an array length = 4 of [1,2] tensors
//   .sort((a, b) => {
//       return a.get(0)>b.get(0)? 1:-1
//   })
//     .slice(0,k)
//     .reduce((acc, pair) => {
//         return  (acc + pair.get(1) )
//     },0)/k
// // calculations.print();
// calculations[0].print(); //?
// calculations.length; //?
