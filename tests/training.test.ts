import { MLP, Value } from '../src';

describe('Neural Network Training', () => {
  test('simple binary classification training', () => {
    const dataset = [
      { inputs: [2.0, 3.0, -1.0], target: 1.0 },  // Class 1
      { inputs: [3.0, -1.0, 0.5], target: 1.0 },  // Class 1
      { inputs: [-1.0, -2.0, -1.0], target: 0.0 }, // Class 0
      { inputs: [-2.0, -1.0, -3.0], target: 0.0 }, // Class 0
    ];

    const model = new MLP(3, [4, 4, 1]);
    
    fixNeuronImplementation(model);
    
    //  initial loss
    let initialLoss = 0;
    for (const example of dataset) {
      const inputs = example.inputs.map(x => new Value(x));
      const prediction = model.call(inputs)[0];
      
      // (p - t)^2 = p^2 - 2*p*t + t^2
      const target = new Value(example.target);
      const pSquared = prediction.mul(prediction);
      const negTwoTimesPTimeT = prediction.mul(target).mul(new Value(-2));
      const loss = pSquared.add(negTwoTimesPTimeT);
      initialLoss += loss.data;
    }
    initialLoss /= dataset.length;
    
    const learningRate = 0.01;
    const epochs = 10;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      let batchLoss = 0;
      
      for (const example of dataset) {
        model.zeroGrad();
        
        const inputs = example.inputs.map(x => new Value(x));
        const prediction = model.call(inputs)[0];
        
        const target = new Value(example.target);
        // (p - t)^2 = p^2 - 2*p*t + t^2
        const pSquared = prediction.mul(prediction);
        const negTwoTimesPTimeT = prediction.mul(target).mul(new Value(-2));
        const loss = pSquared.add(negTwoTimesPTimeT);
        batchLoss += loss.data;
        
        loss.backward();
        
        model.parameters().forEach(p => {
          p.data -= learningRate * p.grad;
        });
      }
      
      batchLoss /= dataset.length;
      // console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${batchLoss}`);
    }
    
    let finalLoss = 0;
    for (const example of dataset) {
      const inputs = example.inputs.map(x => new Value(x));
      const prediction = model.call(inputs)[0];
      const target = new Value(example.target);
      const pSquared = prediction.mul(prediction);
      const negTwoTimesPTimeT = prediction.mul(target).mul(new Value(-2));
      const loss = pSquared.add(negTwoTimesPTimeT);
      finalLoss += loss.data;
    }
    finalLoss /= dataset.length;
    
    expect(finalLoss).toBeLessThan(initialLoss);
    
    const posExample = dataset[0].inputs.map(x => new Value(x));
    const negExample = dataset[2].inputs.map(x => new Value(x));
    
    const posPrediction = model.call(posExample)[0].data;
    const negPrediction = model.call(negExample)[0].data;
    
    expect(posPrediction).toBeGreaterThan(negPrediction);
  });
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