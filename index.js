const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
function knn(features, labels, predictionPoint, k){
    const {mean,variance} = tf.moments(features,0)
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5))//?
    return features
        .sub(mean) //standardize features
        .div(variance.pow(0.5)) //standardize features
        .sub(scaledPrediction) // calculating distance from each point to the prediction point
        .pow(2) // calculating distance from each point to the prediction point
        .sum(1) // calculating distance from each point to the prediction point
        .pow(0.5) // calculating distance from each point to the prediction point
        .expandDims(1) // transform the distances to a vertical 1d tensor
        .concat(labels, 1) // concat the prices to the distances, making it a 2d tensor
        .unstack() // the [4, 2] tensor becomes an array length = 4 of [1,2] tensors
        .sort((a, b) => {

            return a.get(0) > b.get(0) ? 1 : -1
        })
        .slice(0, k)
        .reduce((acc, pair) =>  acc + pair.get(1), 0) / k
}

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long", "sqft_lot"],
    labelColumns: ["price"]
  }
);
// testLabels//?

features = tf.tensor(features)
labels = tf.tensor(labels)




testFeatures.forEach((testPoint, index) => {
    const results = knn(features, labels,tf.tensor(testPoint),10)
    const err = (testLabels[index][0] - results)/ testLabels[index][0]
    console.log("error ", err*100)
})

