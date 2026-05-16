// converted from Karpathy's micrograd engine.py
// https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py 

// p derivs, treat the not focused vars as constants

class Value {
    data: number;
    grad: number;
    private _backward: () => void;
    private _prev: Set<Value>;
    private _op: string;

    constructor(data: number, _children: Value[] = [], _op: string = "") {
        this.data = data;
        this.grad = 0;
        this._backward = () => {};
        this._prev = new Set(_children);
        this._op = _op;
    }

    add(other: Value | number): Value {
        const o = other instanceof Value ? other : new Value(other);
        const out = new Value(this.data + o.data, [this, o], "+");
        // a + b 
        // partial derivatives
        // d(out)/d(a) = 1
        // d(out)/a(b) = 1
        out._backward = () => {
            this.grad += out.grad;
            o.grad += out.grad;
        };
        return out;
    }

    mul(other: Value | number): Value {
        const o = other instanceof Value ? other : new Value(other);
        const out = new Value(this.data * o.data, [this, o], "*");
        // a * b
        // partial detivaitves
        // d(out)/d(a) = b
        // d(out)/d(b) = a`
        // switching the values of a and b below (based on how you do partial derivatives):
        out._backward = () => {
            this.grad += o.data * out.grad;
            o.grad += this.data * out.grad;
        }
        return out;
    }

    pow(other: number): Value {
        const out = new Value(this.data ** other, [this], `**${other}`);
        // use the power rule
        // d(out)/d(a) = n * a^(n-1)
        out._backward = () => {
            this.grad += other * this.data ** (other - 1) * out.grad;
        }
        return out;
    }

    relu(): Value {
        const out = new Value(this.data < 0 ? 0 : this.data, [this], "ReLU");
        return out;
    }

    neg(): Value {
        return this.mul(-1);
    }

    sub(other: Value | number): Value {
        const o = other instanceof Value ? other : new Value(other);
        return this.add(o.neg());
    }

    div(other: Value | number): Value {
        const o = other instanceof Value ? other : new Value(other);
        return this.mul(o.pow(-1));
    }

    // backward pass
    backward(): void {
        // topological sort
        // material for my understanding: https://www.geeksforgeeks.org/dsa/topological-sorting/
        const topo: Value[] = [];
        const visited = new Set<Value>();
        // gives an array where the earliest instantiated Value comes first
        const buildTopo = (v: Value) => {
            if (!visited.has(v)) {
                visited.add(v);
                for (const child of v._prev) {
                    buildTopo(child);
                }
                topo.push(v);
            }
        };
        buildTopo(this);

        this.grad = 1;
        for (let i = topo.length - 1; i >= 0; i--) {
            topo[i]._backward();
        }
    }

    toString(): string {
        return `Value(data=${this.data}, grad=${this.grad})`;
    }
}