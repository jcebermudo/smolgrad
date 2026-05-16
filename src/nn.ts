// converted from Karpathy's nn.py https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py

import { Value } from './engine';

abstract class Module {
    zeroGrad(): void {
        for (const p of this.parameters()) {
            p.grad = 0;
        }
    }

    parameters(): Value[] {
        return [];
    }
}

class Neuron extends Module {
    w: Value[];
    b: Value;
    nonlin: boolean;

    constructor(nin: number, nonlin: boolean = true) {
        super();
        this.w = Array.from({length: nin}, () => new Value(Math.random() * 2 - 1));
        this.b = new Value(0);
        this.nonlin = nonlin;
    }

    call(x: Value[]): Value {
        const act = this.w.reduce(
            (sum, wi, i) => sum.add(wi.mul(x[i])),
            this.b
        );
        return this.nonlin ? act.relu() : act;
        // output is basically w0*x*0 + w1*x1 + ... + wn*xn + b then plug it in to relu if nonlin
    }

    parameters(): Value[] {
        return [...this.w, this.b];
    }

    toString(): string {
        return `${this.nonlin ? "ReLU" : "Linear"}Neuron(${this.w.length})`;
    }
}

class Layer extends Module {
    neurons: Neuron[];

    constructor(nin: number, nout: number, nonlin: boolean = true) {
        super();
        this.neurons = Array.from({ length: nout }, () => new Neuron(nin, nonlin));
    }

    call(x: Value[]): Value | Value[] {
        const out = this.neurons.map((n) => n.call(x));
        return out.length === 1 ? out[0] : out;
    }

    parameters(): Value[] {
        return this.neurons.flatMap((n) => n.parameters());
    }

    toString(): string {
        return `Layer of [${this.neurons.map((n) => n.toString()).join(", ")}]`;
    }
}

class MLP extends Module {
    layers: Layer[];

    constructor(nin: number, nouts: number[]) {
        super();
        const sz = [nin, ...nouts];
        this.layers = nouts.map(
            (_, i) => new Layer(sz[i], sz[i + 1], i !== nouts.length - 1)
        );
    }

    // returns the output of last layer, we do forward pass
    call(x: Value[]): Value | Value[] {
        let current: Value | Value[] = x;
        for (const layer of this.layers) {
            current = layer.call(current as Value[]);
        }
        return current;
    }

    parameters(): Value[] {
        return this.layers.flatMap((layer) => layer.parameters());
    }

    toString(): string {
        return `MLP of [${this.layers.map((l) => l.toString()).join(", ")}]`;
    }
}

export { Module, Neuron, Layer, MLP };