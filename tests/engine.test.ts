import { Value } from '../src';

test('test complex chain of operations', () => {
  const x = new Value(2);
  const y = new Value(3);
  const z = x.mul(y).add(x.pow(2)).relu();

  z.backward();

  expect(z.data).toBe(10);
  expect(x.grad).toBe(7);
  expect(y.grad).toBe(2);
});
