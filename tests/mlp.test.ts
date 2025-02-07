import { MLP, Value } from '../src';

test('test MLP backward propagation', () => {
  const model = new MLP(3, [5, 5, 2]);

  const input = [new Value(0.5), new Value(-1.2), new Value(2.3)];
  const output = model.call(input);

  output[0].backward();

  const params = model.parameters();

  let gradNonZero = params.some(param => Math.abs(param.grad) > 0);
  expect(gradNonZero).toBe(true);
});
