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
        const [B, O] = this.shape;
        const [,O2] = other.shape;

        // broadcasting: allows ops between tensors of diff shapes by automatically expanding smalelr tensors to match the larger one
        // if other has shape [1, O2] while "this" has shape [B, O2] where B > 1
        const broadcast = other.shape[0] === 1 && B > 1;
        for (let b = 0; b < B; b++)
            for(let o = 0; o < O; o++)
                outData[b*O+o] = this.data[b * O + o] + other.data[(broadcast ? 0 : b) * O2 + o];
        const out = new Tensor(outData, this.shape, [this, other], "+");
        out._backward = () => {
            for (let b = 0; b < B; b++)
                for (let o = 0; o < O; o++) {
                    // takes the gradient from the output and adds it to the grad of first tensor
                    this.grad[b*O+o] += out.grad[b*O+o];
                    // if other was broadcasted, all batch elements' grad contribute to the single row at index 0 (it gets summed)
                    other.grad[(broadcast ? 0 : b) * O2 + o] += out.grad[b * O + o]; 
                }
        }
        return out;
    }

    mul(scalar: number): Tensor {
        const outData = this.data.map(v => v * scalar);
        const out = new Tensor(outData, this.shape, [this], "*");
        out._backward = () => {
            for (let i = 0; i < out.grad.length; i++)
                this.grad[i] += scalar * out.grad[i];
        }
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
        out._backward = () => {
            // d(loss)/d(input) = d(Loss)/d(out) @ weight.T
            for (let b = 0; b < B; b++)
                for (let i = 0; i < I; i++)
                    for (let o = 0; o < O; o++)
                        this.grad[b * I + i] += out.grad[b * O + o] * weight.data[i * O + o];

            // d(Loss)/d(weight) = input.T @ d(Loss)/d(out)                                    
              for (let i = 0; i < I; i++)                                                        
                  for (let o = 0; o < O; o++)                                                    
                      for (let b = 0; b < B; b++)
                          weight.grad[i * O + o] += this.data[b * I + i] * out.grad[b * O + o];
        };
        return out;
    }

    relu(): Tensor {
        const outData = this.data.map(v => Math.max(0, v));
        const out = new Tensor(outData, this.shape, [this], "ReLU");
        out._backward = () => {
            for (let i = 0; i < out.grad.length; i++)
                this.grad[i] += (out.data[i] > 0 ? 1 : 0) * out.grad[i];
        }
        return out
    }

    // for cross-entrop loss
    log(): Tensor {
        const outData = this.data.map(v => Math.log(v + 1e-8)); // 1e-8 prevents log(0)
        const out = new Tensor(outData, this.shape, [this], "log");
        out._backward = () => {
            for (let i = 0; i < out.grad.length; i++)
                this.grad[i] += (1 / (this.data[i] + 1e-8)) * out.grad[i];
        };
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
        out._backward = () => {
            // d(softmax)/d(x_i) = softmax_i * (1 - softmax_i) -- diagonal
            // softmax_i * (delta_ij - softmax_j) -- full jacobian
            // simplified when combined w c-e: grad = softmax - one_hot
            // we do the full version (the simplification happens in the loss)
            for(let b = 0; b < B; b++)
                for(let i = 0; i < O; i++){
                    let dot = 0;
                    for (let j = 0; j < O; j++) dot += out.grad[b * O + j] * out.data[b * O +  j];
                    this.grad[b * O + i] += out.data[b * O + i] * (out.grad[b * O + i] - dot); 
                }
        };
        return out;
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