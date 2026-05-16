// converted from Karpathy's micrograd engine.py
// https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py 

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
        return out;
    }

    mul(other: Value | number): Value {
        const o = other instanceof Value ? other : new Value(other);
        const out = new Value(this.data * o.data, [this, o], "*");
        return out;
    }

    pow(other: number): Value {
        const out = new Value(this.data ** other, [this], `**${other}`);
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

    // todo bp


}