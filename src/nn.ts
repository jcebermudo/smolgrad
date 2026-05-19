// converted from Karpathy's nn.py https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py

import { Tensor } from './engine';

abstract class Module {
    zeroGrad(): void {
        for (const p of this.parameters()) {
            p.grad.fill(0)
        }
    }

    parameters(): Tensor[] {
        return [];
    }
}

class Linear extends Module {
    weight: Tensor; // shape [in, out]
    bias: Tensor;   // shape [1, out]
    nonlin: boolean;

    constructor(nin: number, nout: number, nonlin: boolean = true) {
        super();
        // Xavier initialization: keeps variance stable across layers // https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
        const scale = Math.sqrt(2 / nin);
        this.weight = new Tensor(
            Float32Array.from({ length: nin * nout }, () => (Math.random() * 2 - 1) * scale),
            [nin, nout]
        );
        this.bias = new Tensor(new Float32Array(nout), [1, nout]);
        this.nonlin = nonlin;
    }

    call(x: Tensor): Tensor {
        const act = x.matmul(this.weight).add(this.bias);
        return this.nonlin ? act.relu() : act;
    }

    parameters(): Tensor[] {
        return [this.weight, this.bias];
    }

    toString(): string {
        return `${this.nonlin ? "ReLU" : "Linear"}(${this.weight.shape[0]} -> ${this.weight.shape[1]})`;
    }
}

class MLP extends Module {
    layers: Linear[];

    constructor(nin: number, nouts: number[]) {
          super();                                                                               
          const sz = [nin, ...nouts];
          this.layers = nouts.map(
              (_, i) => new Linear(sz[i], sz[i + 1], i !== nouts.length - 1)
          );                                                                                     
      }

    // returns the output of last layer, we do forward pass
    call(x: Tensor): Tensor {
        let current = x;
        for (const layer of this.layers) {
            current = layer.call(current);
        }
        return current;
    }

    parameters(): Tensor[] {
        return this.layers.flatMap((layer) => layer.parameters());
    }

    toString(): string {
        return `MLP of [${this.layers.map((l) => l.toString()).join(", ")}]`;
    }
}

export { Module, Linear, MLP };