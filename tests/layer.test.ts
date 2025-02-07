import { Layer, Value } from '../src';

test('test Layer forward pass', () => {
    const layer = new Layer(3, 2);
    const inputs = [new Value(1), new Value(-1), new Value(0.5)];
  
    const outputs = layer.call(inputs);
  
    expect(outputs.length).toBe(2);
    expect(outputs[0]).toBeInstanceOf(Value);
    expect(outputs[1]).toBeInstanceOf(Value);
  });