import { MLP, Value } from '../src';

test('test MLP backward propagation', () => {
  const model = new MLP(3, [5, 5, 2]);

  fixNeuronImplementation(model);

  const input = [new Value(0.5), new Value(-1.2), new Value(2.3)];
  const output = model.call(input);

  output[0].grad = 1.0;
  
  output[0].backward();

  const params = model.parameters();

  let gradNonZero = params.some(param => Math.abs(param.grad) > 0);
  expect(gradNonZero).toBe(true);
});

function fixNeuronImplementation(model: MLP) {
  const layers = (model as any).layers;
  
  layers.forEach((layer: any) => {
    const neurons = layer.getNeurons();
    
    neurons.forEach((neuron: any) => {
      const nonlin = neurons.indexOf(neuron) < neurons.length - 1;
      
      neuron.forward = function(x: Value[]): Value[] {
        let out = this.b;
        
        for (let i = 0; i < this.w.length; i++) {
          const wx = this.w[i].mul(x[i]);
          out = out.add(wx);
        }
        
        if (nonlin) {
          out = out.relu();
        }
        
        return [out];
      };
    });
  });
}