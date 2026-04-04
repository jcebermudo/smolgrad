import { Tensor, Operation, shapeSize } from './engine'

// MSE loss
// pred, target: same shape — [N] for single sample, [batch, N] for batched
export function mseLoss(pred: Tensor, target: Tensor): Tensor {
  const n = shapeSize(pred.shape);
  const scale = Tensor.scalar(1 / n);
  const neg   = Tensor.scalar(-1);

  const diff = pred.add(target.mul(neg));  // pred - target
  const sq   = diff.mul(diff);             // (pred - target)^2
  return sq.sum().mul(scale);              // mean
}