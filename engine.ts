// computes the total number of elements from a shape array
function shapeSize(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

// broadcasting: a way to make arrays of different shapes work together in ops like add and mul
// instead of copying data, smaller arrays are "virtually" expanded
// takes two shape arrays and returns the resulting shape after broadcasting them together
// this is just a compatability check + calculator
function broadcastShapes(a: number[], b: number[]): number[] {
  const out: number[] = [];
  const maxLen = Math.max(a.length, b.length);
  for (let i = 0; i < maxLen; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    if (da !== db && da !== 1 && db !== 1)
      throw new Error(`Cannot broadcast ${a} and ${b}`);
    out.unshift(Math.max(da, db));
  }
  return out;
}

// for backprop for broadcasted operations
// reduces a gradient back to a smaller shape after broadcasting
// gradients must match the original parameter's shape
function sumToShape(
  grad: Float32Array,
  gradShape: number[],
  targetShape: number[],
): Float32Array {
  const padded = Array(gradShape.length - targetShape.length)
    .fill(1)
    .concat(targetShape);
  let result = grad;
  let currentShape = [...gradShape];

  for (let axis = currentShape.length - 1; axis >= 0; axis--) {
    if (padded[axis] === 1 && currentShape[axis] !== 1) {
      result = sumAlongAxis(result, currentShape, axis);
      currentShape[axis] = 1;
    }
  }

  return result.slice(0, shapeSize(targetShape));
}

// sums a flattened multi-d array along a specified axis
// the multi-d array is flattened but the shape is used to tell how to interpret it
// the axis: dimensions along which you want to sum
// out/in sizes: helps calculate the correct indices in the 1d array
function sumAlongAxis(
  data: Float32Array,
  shape: number[],
  axis: number,
): Float32Array {
  const outerSize = shape.slice(0, axis).reduce((a, b) => a * b, 1);
  const axisSize = shape[axis];
  const innerSize = shape.slice(axis + 1).reduce((a, b) => a * b, 1);
  const out = new Float32Array(outerSize * innerSize);
  for (let o = 0; o < outerSize; o++)
    for (let a = 0; a < axisSize; a++)
      for (let i = 0; i < innerSize; i++)
        out[o * innerSize + i] +=
          data[o * axisSize * innerSize + a * innerSize + i];
  return out;
}

type Operation =
  | "+"
  | "*"
  | "matmul"
  | "tanh"
  | "relu"
  | "exp"
  | "log"
  | "sum"
  | "reshape"
  | "T"
  | "broadcast"
  | "";

class Tensor {
  data: Float32Array;
  grad: Float32Array;
  shape: number[];
  _prev: Set<Tensor>;
  _op: Operation;
  _backward: () => void;

  constructor(data: Float32Array | number[], shape: number[]) {
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.grad = new Float32Array(this.data.length); // zero-initialised
    this.shape = shape;
    this._prev = new Set();
    this._op = "";
    this._backward = () => {};
  }

  static scalar(v: number): Tensor {
    return new Tensor([v], []); // shape=[] means 0-d scalar
  }

  static zeros(shape: number[]): Tensor {
    return new Tensor(new Float32Array(shapeSize(shape)), shape);
  }

  static randn(shape: number[]): Tensor {
    const n = shapeSize(shape);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      // box-muller transfor for normal
      const u = 1 - Math.random(),
        v = Math.random();
      data[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    return new Tensor(data, shape);
  }

  // ops

  add(other: Tensor): Tensor {
    const outShape = broadcastShapes(this.shape, other.shape);
    const n = shapeSize(outShape);
    const out = new Tensor(new Float32Array(n), outShape);

    // broadcast-aware forward
    for (let i = 0; i < n; i++) {
      out.data[i] = this._bget(i, outShape) + other._bget(i, outShape);
    }

    out._prev = new Set([this, other]);
    out._op = "+";
    out._backward = () => {
      const dA = sumToShape(out.grad, outShape, this.shape);
      const dB = sumToShape(out.grad, outShape, other.shape);
      for (let i = 0; i < this.grad.length; i++) this.grad[i] += dA[i];
      for (let i = 0; i < other.grad.length; i++) other.grad[i] += dB[i];
    };
    return out;
  }

  mul(other: Tensor): Tensor {
    const outShape = broadcastShapes(this.shape, other.shape);
    const n = shapeSize(outShape);
    const out = new Tensor(new Float32Array(n), outShape);

    for (let i = 0; i < n; i++) {
      out.data[i] = this._bget(i, outShape) * other._bget(i, outShape);
    }

    out._prev = new Set([this, other]);
    out._op = "*";
    out._backward = () => {
      const dA = new Float32Array(n);
      const dB = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        dA[i] = other._bget(i, outShape) * out.grad[i];
        dB[i] = this._bget(i, outShape) * out.grad[i];
      }
      const sdA = sumToShape(dA, outShape, this.shape);
      const sdB = sumToShape(dB, outShape, other.shape);
      for (let i = 0; i < this.grad.length; i++) this.grad[i] += sdA[i];
      for (let i = 0; i < other.grad.length; i++) other.grad[i] += sdB[i];
    };
    return out;
  }

  matmul(other: Tensor): Tensor {
    const [M, K] = this.shape;
    const [K2, N] = other.shape;
    if (K !== K2)
      throw new Error(`matmul shape mismatch: ${this.shape} @ ${other.shape}`);

    const out = Tensor.zeros([M, N]);
    for (let m = 0; m < M; m++)
      for (let n = 0; n < N; n++)
        for (let k = 0; k < K; k++)
          out.data[m * N + n] += this.data[m * K + k] * other.data[k * N + n];

    out._prev = new Set([this, other]);
    out._op = "matmul";
    out._backward = () => {
      // dL/dA = dL/dOut @ B.T    shape [M,N] @ [N,K] → [M,K]
      for (let m = 0; m < M; m++)
        for (let k = 0; k < K; k++)
          for (let n = 0; n < N; n++)
            this.grad[m * K + k] += out.grad[m * N + n] * other.data[k * N + n];

      // dL/dB = A.T @ dL/dOut   shape [K,M] @ [M,N] → [K,N]
      for (let k = 0; k < K; k++)
        for (let n = 0; n < N; n++)
          for (let m = 0; m < M; m++)
            other.grad[k * N + n] += this.data[m * K + k] * out.grad[m * N + n];
    };
    return out;
  }

  tanh(): Tensor {
    const out = Tensor.zeros(this.shape);
    for (let i = 0; i < this.data.length; i++)
      out.data[i] = Math.tanh(this.data[i]);
    out._prev = new Set([this]);
    out._op = "tanh";
    out._backward = () => {
      for (let i = 0; i < this.grad.length; i++)
        this.grad[i] += (1 - out.data[i] ** 2) * out.grad[i];
    };
    return out;
  }

  relu(): Tensor {
    const out = Tensor.zeros(this.shape)
    for (let i = 0; i < this.data.length; i++) out.data[i] = Math.max(0, this.data[i])
    out._prev = new Set([this])
    out._op   = 'relu'
    out._backward = () => {
      for (let i = 0; i < this.grad.length; i++)
        this.grad[i] += (this.data[i] > 0 ? 1 : 0) * out.grad[i]
    }
    return out
  }

  exp(): Tensor {
    const out = Tensor.zeros(this.shape)
    for (let i = 0; i < this.data.length; i++) out.data[i] = Math.exp(this.data[i])
    out._prev = new Set([this])
    out._op   = 'exp'
    out._backward = () => {
      for (let i = 0; i < this.grad.length; i++)
        this.grad[i] += out.data[i] * out.grad[i]
    }
    return out
  }

  log(): Tensor {
    const out = Tensor.zeros(this.shape)
    for (let i = 0; i < this.data.length; i++) out.data[i] = Math.log(this.data[i])
    out._prev = new Set([this])
    out._op   = 'log'
    out._backward = () => {
      for (let i = 0; i < this.grad.length; i++)
        this.grad[i] += (1 / this.data[i]) * out.grad[i]
    }
    return out
  }

  sum(axis?: number): Tensor {
    if (axis === undefined) {
      // full sum → scalar
      const s = this.data.reduce((a, b) => a + b, 0)
      const out = new Tensor([s], [])
      out._prev = new Set([this])
      out._op   = 'sum'
      out._backward = () => {
        for (let i = 0; i < this.grad.length; i++) this.grad[i] += out.grad[0]
      }
      return out
    }
    // axis sum
    const newShape = this.shape.filter((_, i) => i !== axis)
    const out = Tensor.zeros(newShape.length ? newShape : [1])
    const reduced = sumAlongAxis(this.data, this.shape, axis)
    out.data.set(reduced)
    out._prev = new Set([this])
    out._op   = 'sum'
    out._backward = () => {
      // broadcast grad back along the summed axis
      const axisSize  = this.shape[axis]
      const innerSize = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1)
      const outerSize = this.shape.slice(0, axis).reduce((a, b) => a * b, 1)
      for (let o = 0; o < outerSize; o++)
        for (let a = 0; a < axisSize; a++)
          for (let i = 0; i < innerSize; i++)
            this.grad[o * axisSize * innerSize + a * innerSize + i] +=
              out.grad[o * innerSize + i]
    }
    return out
  }

  reshape(newShape: number[]): Tensor {
    if (shapeSize(newShape) !== shapeSize(this.shape))
      throw new Error('reshape: size mismatch')
    const out   = new Tensor(this.data, newShape)   // shares buffer
    out._prev   = new Set([this])
    out._op     = 'reshape'
    out._backward = () => {
      for (let i = 0; i < this.grad.length; i++) this.grad[i] += out.grad[i]
    }
    return out
  }

  get T(): Tensor {
    const [R, C] = this.shape
    const out    = Tensor.zeros([C, R])
    for (let r = 0; r < R; r++)
      for (let c = 0; c < C; c++)
        out.data[c * R + r] = this.data[r * C + c]
    out._prev = new Set([this])
    out._op   = 'T'
    out._backward = () => {
      for (let r = 0; r < R; r++)
        for (let c = 0; c < C; c++)
          this.grad[r * C + c] += out.grad[c * R + r]
    }
    return out
  }

  // backward pass
  backward(): void {
    const topo:    Tensor[] = []
    const visited           = new Set<Tensor>()
    const build = (t: Tensor) => {
      if (!visited.has(t)) {
        visited.add(t)
        for (const child of t._prev) build(child)
        topo.push(t)
      }
    }
    build(this)

    this.grad.fill(1)                               // dL/dL = 1
    for (const t of topo.reverse()) t._backward()
  }

  zeroGrad(): void { this.grad.fill(0) }

  private _bget(flatIdx: number, outShape: number[]): number {
    // convert flat index in outShape to flat index in this.shape
    const coords: number[] = []
    let rem = flatIdx
    for (let i = outShape.length - 1; i >= 0; i--) {
      coords.unshift(rem % outShape[i])
      rem = Math.floor(rem / outShape[i])
    }
    // align coords to this.shape (left-pad with 0)
    const pad    = outShape.length - this.shape.length
    let myFlat   = 0
    let stride   = 1
    for (let i = this.shape.length - 1; i >= 0; i--) {
      const c = coords[i + pad]
      myFlat += (this.shape[i] === 1 ? 0 : c) * stride
      stride *= this.shape[i]
    }
    return this.data[myFlat]
  }

  toString(): string {
    return `Tensor(shape=[${this.shape}], data=[${Array.from(this.data).slice(0, 6)}${this.data.length > 6 ? '...' : ''}])`
  }
}
