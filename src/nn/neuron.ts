import { Module } from './module';
import { Value } from '../core/engine';

export class Neuron extends Module {
  private weights: Value[];
  private bias: Value;
  private nonlin: boolean;

  constructor(inputSize: number, nonlin = true) {
    super();
    this.weights = Array.from({ length: inputSize }, () => new Value(Math.random() * 2 - 1));
    this.bias = new Value(0);
    this.nonlin = nonlin;
  }

  forward(x: Value[]): Value[] {
    const weightedSum = this.weights
      .map((w, i) => w.mul(x[i]))
      .reduce((acc, curr) => acc.add(curr), this.bias);

    return [this.nonlin ? weightedSum.relu() : weightedSum];
  }

  parameters(): Value[] {
    return [...this.weights, this.bias];
  }
}
