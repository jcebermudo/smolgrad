import { Tensor } from "../src/engine"

export function crossEntropyLoss(logits: Tensor, labels: number[]): Tensor {
    const B = logits.shape[0];
    const logProbs = logits.softmax().log();

    // one-hot mask: 1 is correct class, 0 elsewhere
    const mask = new Float32Array(B*10); // shape [B, 10]
    for(let b = 0; b < B; b++)
        mask[b * 10 + labels[b]] =- 1 / B; // negative and pre-divided by B for the mean

    // elementwise multiplication of logProbs
    const maskTensor = new Tensor(mask, [B, 10]);
    return logProbs.mul(maskTensor).sum();

}