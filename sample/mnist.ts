import { MLP } from "../src/nn"
import { crossEntropyLoss } from "../src/loss"
import { loadMNIST } from "../src/loader"

// mapping:
// 784 inputs -> 128 -> 64 -> 10 outputs
const model = new MLP(784, [128, 64, 10]);
const mnist = loadMNIST("./data");

const lr = 0.01;
const batchSize = 32;
const epochs = 5;

for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;

    const numBatches = Math.floor(mnist.trainSize / batchSize);

    for (let i = 0; i < mnist.trainSize; i += batchSize) {
        const [inputs, labels] = mnist.getTrainBatch(i, batchSize);

        // forward pass
        const logits = model.call(inputs);
        const loss = crossEntropyLoss(logits, labels);

        // backward pass
        model.zeroGrad();
        loss.backward();

        // SGD
        for (const p of model.parameters()) {
            for (let j = 0; j < p.data.length; j++)
                p.data[j] -= lr * p.grad[j];
        }

        totalLoss += loss.data[0];

        const batch = i / batchSize;
        if (batch % 100 === 0)                                                                                                                                        
            console.log(`  epoch ${epoch + 1} | batch ${batch}/${numBatches} | loss: ${loss.data[0].toFixed(4)}`);
    }
    console.log(`epoch ${epoch + 1} done, avg loss: ${(totalLoss / numBatches).toFixed(4)}`);
}