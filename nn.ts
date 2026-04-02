abstract class Module {
    zeroGrad(): void {
        for (const p of this.parameters()) p.grad = 0
    }
    parameters(): Value[] {
     return []
    }
}

class Neuron extends Module {
    w: Value[]
    b: Value
    nonlin: boolean

    constructor(nin: number, nonlin = true) {
        super()
        this.w = Array.from({length: nin}, () => new Value(Math.random() * 2 -1))
        this.b = new Value(0)
        this.nonlin = nonlin
    }

    call(x: Value[]): Value {
        const act = this.w.reduce((sum, wi, i) => sum.add(wi.mul(x[i])), this.b)
        return this.nonlin ? act.tanh() : act
    }

    parameters(): Value[] {
        return [...this.w, this.b]
    }

    toString(): string {
        return `${this.nonlin ? 'ReLU' : 'Linear'}Neuron(${this.w.length})`
    }
}