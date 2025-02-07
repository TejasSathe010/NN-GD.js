import { Module } from './module';
import { Neuron } from './neuron';
import { Value } from '../core/engine';

export class Layer extends Module {
  private neurons: Neuron[];

  constructor(inputSize: number, outputSize: number, nonlin = true) {
    super();
    this.neurons = Array.from({ length: outputSize }, () => new Neuron(inputSize, nonlin));
  }

  forward(x: Value[]): Value[] {
    return this.neurons.flatMap(neuron => neuron.forward(x));
  }

  parameters(): Value[] {
    return this.neurons.flatMap(neuron => neuron.parameters());
  }
}
