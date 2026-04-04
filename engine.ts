// helpers
function shapeSize(shape: number[]): number {
    return shape.reduce((a, b) => a * b, 1)
}

function broadcastShapes(a: number[], b: number[]): number[] {
    const out: number[] = []
    const maxLen = Math.max(a.length, b.length)
    for (let i = 0; i < maxLen; i++) {
        const da = a[a.length - 1 - i] ?? 1
        const db = b[b.length - 1 - i] ?? 1
        if (da !== db && da !== 1 && db !== 1)
            throw new Error(`Cannot broadcast ${a} and ${b}`)
        out.unshift(Math.max(da, db))
    }
    return out
}


// undo broadcasting
function sumToShape(grad: Float32Array, gradShape: number[], targetShape: number[]): Float32Array {
    const padded = Array(gradShape.length - targetShape.length).fill(1).concat(targetShape)
    let result = grad
    let currentShape = [...gradShape]

    for (let axis = currentShape.length - 1; axis >= 0; axis--) {
        if(padded[axis] === 1 && currentShape[axis] !== 1) {
            result = sumAlongAxis(result, currentShape, axis)
            currentShape[axis] = 1
        }
    }

    return result.slice(0, shapeSize(targetShape))
}

function sumAlongAxis(data: Float32Array, shape: number[], axis: number): Float32Array {
  const outerSize = shape.slice(0, axis).reduce((a, b) => a * b, 1)
  const axisSize  = shape[axis]
  const innerSize = shape.slice(axis + 1).reduce((a, b) => a * b, 1)
  const out = new Float32Array(outerSize * innerSize)
  for (let o = 0; o < outerSize; o++)
    for (let a = 0; a < axisSize; a++)
      for (let i = 0; i < innerSize; i++)
        out[o * innerSize + i] += data[o * axisSize * innerSize + a * innerSize + i]
  return out
}


type Operation = '+' | '*' | 'matmul' | 'tanh' | 'relu' | 'exp' | 'log' |
                 'sum' | 'reshape' | 'T' | 'broadcast' | ''

class Value {
    data: number;
    grad: number;
    _prev: Set<Value>
    _op: Operation
    label: string
    private _backward: () => void

    constructor(data: number, _children: Value[] = [], _op: Operation = '', label = '') {
        this.data = data;
        this.grad = 0;
        this._prev = new Set(_children);
        this._op = _op
        this.label = label
        this._backward = () => {}
    }

    // ops

    add(other: Value | number): Value {
        const o = other instanceof Value ? other : new Value(other)
        const out = new Value(this.data + o.data, [this, o], '+')
        out._backward = () => {
            this.grad += out.grad
            o.grad += out.grad
        }
        return out
    }

    mul(other: Value | number): Value {
        const o = other instanceof Value ? other : new Value(other)
        const out = new Value(this.data * o.data, [this, o], '*')
        out._backward = () => {
            this.grad += out.grad * o.data
            o.grad += this.data * out.grad
        }
        return out
    }

    tanh(): Value {
        const t = Math.tanh(this.data)
        const out = new Value(t, [this], 'tanh')
        out._backward = () => {
            this.grad += (1 - t ** 2) * out.grad // sech^2(x)
        }
        return out
    }

    relu(): Value {
        const out = new Value(Math.max(0, this.data), [this], 'relu')
        out._backward = () => {
            this.grad += (this.data > 0 ? 1 : 0) * out.grad
        }
        return out
    }

    exp(): Value {
        const e = Math.exp(this.data)
        const out = new Value(e, [this], 'exp')
        out._backward = () => {
            this.grad += e * out.grad
        }
        return out
    }

    pow(n: number): Value {
        const out = new Value(this.data ** n, [this], '' as Operation)
        out._backward = () => {
            this.grad += n * this.data ** (n-1) * out.grad
        }
        return out
    }

    backward(): void {
        // topo sort to process nodes in order
        const topo: Value[] = []
        const visited = new Set<Value>()
        const build = (v: Value) => {
            if (!visited.has(v)) {
                visited.add(v)
                for (const child of v._prev) build(child)
                topo.push(v)
            }
        }
        build(this)

        this.grad = 1 // derivatve with respect to itself is 1
        for (const v of topo.reverse()) v._backward()
    }

neg(): Value {
    return this.mul(-1)
}

sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other)
    return this.add(o.neg())
}

rsub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other)
    return o.add(this.neg())
}

radd(other: Value | number): Value {
    return this.add(other)
}

rmul(other: Value | number): Value {
    return this.mul(other)
}

div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other)
    return this.mul(o.pow(-1))
}

rdiv(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other)
    return o.mul(this.pow(-1))
}

toString(): string {
    return `Value(data=${this.data}, grad=${this.grad})`
}
 }