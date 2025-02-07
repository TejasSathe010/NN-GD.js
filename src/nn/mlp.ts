import { Module } from './module';
import { Layer } from './layer';
import { Value } from '../core/engine';

export class MLP extends Module {
  private layers: Layer[];

  constructor(inputSize: number, layerSizes: number[]) {
    super();
    this.layers = layerSizes.map((size, i) => new Layer(i === 0 ? inputSize : layerSizes[i - 1], size, i < layerSizes.length - 1));
  }

  forward(x: Value[]): Value[] {
    return this.layers.reduce((input, layer) => layer.forward(input), x);
  }

  parameters(): Value[] {
    return this.layers.flatMap(layer => layer.parameters());
  }

  zeroGrad(): void {
    this.parameters().forEach(param => {
      param.grad = 0;
    });
  }
  
}
