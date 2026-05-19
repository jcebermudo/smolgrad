// converted from Karpathy's micrograd engine.py
// https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py 

// p derivs, treat the not focused vars as constants

export class Tensor {
    data: Float32Array;
    grad: Float32Array;
    shape: [number, number];
    private _backward: () => void;
    private _prev: Set<Tensor>;
    private _op: string;

    constructor(data: Float32Array, shape: [number, number], _children: Tensor[] = [], _op: string = "") {
        this.data = data;
        this.grad = new Float32Array(data.length);
        this._backward = () => {};
        this._prev = new Set(_children);
        this._op = _op;
    }

    add(other: Tensor): Tensor {
        const outData = new Float32Array(this.data.length);
        for (let i = 0; i < this.data.length; i++)
            outData[i] = this.data[i] + other.data[i];
        const out = new Tensor(outData, this.shape, [this, other], "+");
        return out
    }

    mul(scalar: number): Tensor {
        const outData = this.data.map(v => v * scalar);
        const out = new Tensor(outData, this.shape, [this], "*");
        return out
    }

    matmul(weight: Tensor): Tensor {
        // B: batch size
        const [B, I] = this.shape;
        // O: output size 
        const [,O] = weight.shape;

        // batch size * output
        const outData = new Float32Array(B * O);

        // manual matrix multiplication element by element
        for(let b = 0; b < B; b++) // each sample in a batch
            for (let o = 0; o < O; o++) // each output neuron
                for(let i = 0; i < I; i++) // each input feat.
                    // output[b, o] = Σ(input[b, i] × weight[i, o])  for i = 0 to I-1
                    // this.data[b * I + i] gets the input value for sample b, feature i
                    // weight.data[i * O + o] get the weight from input feature i to output neuron o
                    // outData[b * O + o] stores the result for sample b out o
                    // outData: collection of all output neuron results
                    outData[b * O + o] += this.data[b * I + i] * weight.data[i * O + o]

        const out = new Tensor(outData, [B, O], [this, weight], "matmul")
        return out;
    }

    relu(): Tensor {
        const outData = this.data.map(v => Math.max(0, v));
        const out = new Tensor(outData, this.shape, [this], "ReLU");
        return out
    }

    // for cross-entrop loss
    log(): Tensor {
        const outData = this.data.map(v => Math.log(v + 1e-8)); // 1e-8 prevents log(0)
        const out = new Tensor(outData, this.shape, [this], "log");
        return out;
    }

    // for classif output
    softmax(): Tensor {
        const [B, O] = this.shape;
        const outData = new Float32Array(B*O);
        for (let b = 0; b < B; b++) {
            // sub max per row for num stability
            let max = -Infinity;
            for (let o = 0; o < O; o++) max = Math.max(max, this.data[b*O+o]);

            let sum = 0;
            for (let o = 0; o < O; o++) {
                outData[b * O + o] = Math.exp(this.data[b * O + o] - max);
                sum += outData[b * O + o];
            }
            for (let o = 0; o < O; o++) outData[b * O + o] /= sum;
        }
        const out = new Tensor(outData, this.shape, [this], "softmax");
        return out
    }

    // backward pass
    backward(): void {
        // topological sort
        // material for my understanding: https://www.geeksforgeeks.org/dsa/topological-sorting/
        const topo: Tensor[] = [];
        const visited = new Set<Tensor>();
        // gives an array where the earliest instantiated Value comes first
        const buildTopo = (v: Tensor) => {
            if (!visited.has(v)) {
                visited.add(v);
                for (const child of v._prev) {
                    buildTopo(child);
                }
                topo.push(v);
            }
        };
        buildTopo(this);

        this.grad.fill(1);
        for (let i = topo.length - 1; i >= 0; i--) {
            topo[i]._backward();
        }
    }

    toString(): string {
        return `Tensor(shape=[${this.shape}], data=[${this.data}], grad=[${this.grad}])`;
    }
}