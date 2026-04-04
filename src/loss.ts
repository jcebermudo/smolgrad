import { Tensor, Operation, shapeSize } from "./engine";

// MSE loss
// pred, target: same shape — [N] for single sample, [batch, N] for batched
export function mseLoss(pred: Tensor, target: Tensor): Tensor {
  const n = shapeSize(pred.shape);
  const scale = Tensor.scalar(1 / n);
  const neg = Tensor.scalar(-1);

  const diff = pred.add(target.mul(neg)); // pred - target
  const sq = diff.mul(diff); // (pred - target)^2
  return sq.sum().mul(scale); // mean
}

// log softmax
// logits: [batch, C]  →  log-probabilities: [batch, C]
// kept unexported — callers should use crossEntropyLoss
function logSoftmax(logits: Tensor): Tensor {
  const [B, C] = logits.shape;

  // row-wise max for numerical stability
  const maxVals = new Float32Array(B);
  for (let b = 0; b < B; b++) {
    let m = -Infinity;
    for (let c = 0; c < C; c++) m = Math.max(m, logits.data[b * C + c]);
    maxVals[b] = m;
  }

  // shifted logits
  const expData = new Float32Array(B * C);
  const rowSums = new Float32Array(B);
  for (let b = 0; b < B; b++) {
    for (let c = 0; c < C; c++) {
      expData[b * C + c] = Math.exp(logits.data[b * C + c] - maxVals[b]);
      rowSums[b] += expData[b * C + c];
    }
  }

  // log_softmax[b,c] = (logit - max) - log(rowSum)
  const outData = new Float32Array(B * C);
  for (let b = 0; b < B; b++)
    for (let c = 0; c < C; c++)
      outData[b * C + c] =
        logits.data[b * C + c] - maxVals[b] - Math.log(rowSums[b]);

  const out = new Tensor(outData, [B, C]);
  out._prev = new Set([logits]);
  out._op = "log_softmax";

  // backward: d(lsm_i)/d(logit_j) = δ_ij − softmax_j
  // → dL/d(logit_j) = dL/d(lsm_j) − softmax_j * Σ_i dL/d(lsm_i)

  out._backward = () => {
    for (let b = 0; b < B; b++) {
      let gradSum = 0;
      for (let c = 0; c < C; c++) gradSum += out.grad[b * C + c];

      for (let c = 0; c < C; c++) {
        const sm = expData[b * C + c] / rowSums[b];
        logits.grad[b * C + c] += out.grad[b * C + c] - sm * gradSum;
      }
    }
  };

  return out;
}

// nll loss
// logProbs: [batch, C] log-probabilities    targets: class indices, length batch
// kept unexported — implementation detail of crossEntropyLoss
function nllLoss(logProbs: Tensor, targets: number[]): Tensor {
  const [B, C] = logProbs.shape;

  // pick the log-prob of the correct class for each sample, average
  let total = 0;
  for (let b = 0; b < B; b++) total += logProbs.data[b * C + targets[b]];
  const loss = -total / B;

  const out = new Tensor([loss], []);
  out._prev = new Set([logProbs]);
  out._op   = "";

  // only the correct class slot gets a gradient
  out._backward = () => {
    for (let b = 0; b < B; b++)
      logProbs.grad[b * C + targets[b]] -= out.grad[0] / B;
  };

  return out;
}

// cross entropy loss
export function crossEntropyLoss(logits: Tensor, targets: number[]): Tensor {
  return nllLoss(logSoftmax(logits), targets);
}
