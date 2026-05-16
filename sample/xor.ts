import { Value } from '../src/engine';
import { MLP } from '../src/nn';

// XOR truth table
const data: [number, number, number][] = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
];

// 2 inputs -> 8 hidden -> 8 hidden -> 1 output (linear)
const model = new MLP(2, [8, 8, 1]);

const learningRate = 0.05;
const epochs = 500;

for (let epoch = 0; epoch <= epochs; epoch++) {
    // compute MSE loss over all samples
    let loss = new Value(0);
    for (const [x0, x1, target] of data) {
        const x = [new Value(x0), new Value(x1)];
        const pred = model.call(x) as Value;
        const err = pred.sub(target);
        loss = loss.add(err.pow(2));
    }

    // backward
    model.zeroGrad();
    loss.backward();

    // SGD update
    for (const p of model.parameters()) {
        p.data -= learningRate * p.grad;
    }

    if (epoch % 100 === 0) {
        console.log(`epoch ${epoch}  loss: ${loss.data.toFixed(6)}`);
    }
}

// final predictions
console.log('\nPredictions after training:');
for (const [x0, x1, target] of data) {
    const pred = model.call([new Value(x0), new Value(x1)]) as Value;
    console.log(`[${x0}, ${x1}] -> ${pred.data.toFixed(4)}  (target: ${target})`);
}
