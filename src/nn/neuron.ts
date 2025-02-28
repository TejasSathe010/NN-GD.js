import { Value } from '../core/engine';

export class Neuron {
  public b: Value; 
  public w: Value[]; 
  
  constructor(inputSize: number, nonlin: boolean) {
    this.b = new Value(Math.random() * 2 - 1); 
    this.w = Array.from({ length: inputSize }, () => new Value(Math.random())); 
  }

  forward(x: Value[]): Value[] {
    const weightedSum = this.w.reduce((sum, w, i) => sum + w.data * x[i].data, this.b.data);
    return [new Value(weightedSum)]; 
  }

  parameters(): Value[] {
    return [this.b, ...this.w];
  }
}
