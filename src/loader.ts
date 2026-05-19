import * as fs from 'fs';
import { Tensor } from "../src/engine"

// these are loaders for the MNIST
function loadImages(path: string): Float32Array[] {
    const buf = fs.readFileSync(path);
    const n = buf.readInt32BE(4);
    const pixels = buf.subarray(16);
    const images: Float32Array[] = [];

    for (let i = 0; i < n; i++) {
        const img = new Float32Array(784);
        for (let j = 0; j < 784; j++) 
            img[j] = pixels[i * 784 + j] / 255;
        images.push(img);
    }
    return images;
}

function loadLabels(path: string): number[] {
    const buf = fs.readFileSync(path);
    const n = buf.readInt32BE(4);
    const labels: number[] = [];
    for (let i = 0; i < n; i++)
        labels.push(buf[8 + i]);
    return labels;
}

function getBatch(images: Float32Array[], labels: number[], start: number, batchSize: number):  [Tensor, number[]] {
    const B = Math.min(batchSize, images.length - start);
    const data = new Float32Array(B * 784);

    for (let b = 0; b < B; b++)
        data.set(images[start + b], b * 784);

    return [new Tensor(data, [B, 784]), labels.slice(start, start + B)];
}

export function loadMNIST(dataDir: string) {
    const trainImages = loadImages(`${dataDir}/train-images-idx3-ubyte`);
    const trainLabels = loadLabels(`${dataDir}/train-labels-idx1-ubyte`);
    const testImages = loadImages(`${dataDir}/t10k-images-idx3-ubyte`);
    const testLabels = loadLabels(`${dataDir}/t10k-labels-idx1-ubyte`);

    return {
        trainSize: trainImages.length,
        testSize: testImages.length,
        getTrainBatch: (start: number, batchSize: number) => getBatch(trainImages, trainLabels, start, batchSize),
        getTestBatch: (start: number, batchSize: number) => getBatch(testImages, testLabels, start, batchSize),
    };
}
