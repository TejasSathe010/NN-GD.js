/**
 * The core engine of the autograd system, implementing the Value class
 * for automatic differentiation.
 */

export class Value {
  public data: number;
  public grad: number;
  private _prev: Set<Value>;
  private _op: string;
  private _backward: () => void;

  constructor(data: number, _children: Value[] = [], _op: string = '') {
    this.data = data;
    this.grad = 0;
    this._prev = new Set(_children);
    this._op = _op;
    this._backward = () => {}; // Default no-op backward function
  }

  // Addition operation
  add(other: Value): Value {
    other = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + other.data, [this, other], '+');
    
    out._backward = () => {
      this.grad += out.grad;
      other.grad += out.grad;
    };
    
    return out;
  }

  // Multiplication operation
  mul(other: Value): Value {
    other = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * other.data, [this, other], '*');
    
    out._backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    
    return out;
  }

  // Subtraction (useful for loss functions)
  sub(other: Value): Value {
    other = other instanceof Value ? other : new Value(other);
    return this.add(other.mul(new Value(-1)));
  }

  // Division
  div(other: Value): Value {
    other = other instanceof Value ? other : new Value(other);
    return this.mul(other.pow(-1));
  }

  // Power operation
  pow(exponent: number): Value {
    const out = new Value(Math.pow(this.data, exponent), [this], `^${exponent}`);
    
    out._backward = () => {
      this.grad += (exponent * Math.pow(this.data, exponent - 1)) * out.grad;
    };
    
    return out;
  }

  // ReLU activation function
  relu(): Value {
    const out = new Value(this.data < 0 ? 0 : this.data, [this], 'ReLU');
    
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    
    return out;
  }

  // Sigmoid activation function (useful for binary classification)
  sigmoid(): Value {
    const sig = 1 / (1 + Math.exp(-this.data));
    const out = new Value(sig, [this], 'sigmoid');
    
    out._backward = () => {
      this.grad += (out.data * (1 - out.data)) * out.grad;
    };
    
    return out;
  }

  // Tanh activation function
  tanh(): Value {
    const x = this.data;
    const t = (Math.exp(2*x) - 1) / (Math.exp(2*x) + 1);
    const out = new Value(t, [this], 'tanh');
    
    out._backward = () => {
      this.grad += (1 - t*t) * out.grad;
    };
    
    return out;
  }

  // Backward pass to compute gradients
  backward(): void {
    // Topological sort
    const topo: Value[] = [];
    const visited = new Set<Value>();
    
    function buildTopo(v: Value) {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._prev) {
          buildTopo(child);
        }
        topo.push(v);
      }
    }
    
    buildTopo(this);
    
    // Set gradient of output to 1 if not already set
    this.grad = 1;
    
    // Backward pass in reverse topological order
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }

  // Utility to convert to a string representation
  toString(): string {
    return `Value(data=${this.data}, grad=${this.grad})`;
  }
}