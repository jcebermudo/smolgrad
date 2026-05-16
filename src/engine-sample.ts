const a = new Value(2);
const b = new Value(3);

// c = a + b
const c = a.add(b); // c = 2 + 3 = 5
// d = c * 2
const d = c.mul(2); // d = 5 * 2 = 10
// e = d ^ 2
const e = d.pow(2); // e = 10^2 = 100
// f = relu(e)
const f = e.relu(); // f = relu(100) = 100


f.backward();

// chain rule in action
// f -> d(f)/d(e) = 1
// e -> d(e)/d(d) = 2 * d = 20
// d -> d(d)/d(c) = 2
// c -> d(c)/d(a) = 1
// c -> d(c)/d(a) = 1

// d(f)/d(a) = d(f)/d(e) * d(e)/d(d) * d(d)/d(c) * d(c)/d(a)
// 1 * 20 * 2 * 1 = 40