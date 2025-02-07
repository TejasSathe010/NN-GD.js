import { Value } from '../core/engine';

export abstract class Module {
  abstract parameters(): Value[];

  call(x: Value[] | number[]): Value[] {
    const values = x.map(xi => (xi instanceof Value ? xi : new Value(xi)));
    return this.forward(values);
  }

  abstract forward(x: Value[]): Value[];

  zeroGrad(): void {
    this.parameters().forEach(param => (param.grad = 0));
  }
}
