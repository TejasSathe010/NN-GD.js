import { Value } from '../core/engine';
import { Module } from './module';

export class Neuron extends Module {
  public b: Value;
  public w: Value[];
  private nonlin: boolean;
  
  constructor(inputSize: number, nonlin = true) {
    super();
    const scale = Math.sqrt(2.0 / inputSize);
    this.w = Array.from({ length: inputSize }, () => new Value((Math.random() * 2 - 1) * scale));
    this.b = new Value(0);
    this.nonlin = nonlin;
  }

  forward(x: Value[]): Value[] {
    let out = this.b;
    
    for (let i = 0; i < this.w.length; i++) {
      const wx = this.w[i].mul(x[i]);
      out = out.add(wx);
    }
    
    if (this.nonlin) {
      out = out.relu();
    }
    
    return [out];
  }

  parameters(): Value[] {
    return [this.b, ...this.w];
  }
}