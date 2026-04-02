type Operation = '+' | '*' | 'tanh' | 'relu' | 'exp' | ''

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
 }