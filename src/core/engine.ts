export class Value {
    data: number;
    grad: number;
    children: Value[];
    op: string;
    private _backward: () => void;
  
    constructor(data: number, children: Value[] = [], op = '') {
      this.data = data;
      this.grad = 0;
      this.children = children;
      this.op = op;
      this._backward = () => {};
    }
  
    add(other: Value | number): Value {
      const otherValue = ensureValue(other);
      const result = new Value(this.data + otherValue.data, [this, otherValue], '+');
  
      result._backward = () => {
        this.grad += result.grad;
        otherValue.grad += result.grad;
      };
  
      return result;
    }
  
    mul(other: Value | number): Value {
      const otherValue = ensureValue(other);
      const result = new Value(this.data * otherValue.data, [this, otherValue], '*');
  
      result._backward = () => {
        this.grad += otherValue.data * result.grad;
        otherValue.grad += this.data * result.grad;
      };
  
      return result;
    }
  
    pow(exp: number): Value {
      const result = new Value(Math.pow(this.data, exp), [this], `**${exp}`);
  
      result._backward = () => {
        this.grad += exp * Math.pow(this.data, exp - 1) * result.grad;
      };
  
      return result;
    }
  
    relu(): Value {
      const result = new Value(this.data < 0 ? 0 : this.data, [this], 'ReLU');
  
      result._backward = () => {
        this.grad += result.data > 0 ? 1 * result.grad : 0;
      };
  
      return result;
    }
  
    backward(): void {
      const topo: Value[] = [];
      const visited = new Set<Value>();
  
      const buildTopo = (v: Value) => {
        if (!visited.has(v)) {
          visited.add(v);
          for (const child of v.children) buildTopo(child);
          topo.push(v);
        }
      };
  
      buildTopo(this);
      this.grad = 1;
      topo.reverse().forEach(v => v._backward());
    }
  
    toString(): string {
      return `Value(data=${this.data}, grad=${this.grad}, op=${this.op})`;
    }
  }
  
  export function ensureValue(x: Value | number): Value {
    return x instanceof Value ? x : new Value(x);
  }
  