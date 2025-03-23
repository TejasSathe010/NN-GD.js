import { MLP } from '../nn/mlp';
import { Layer } from '../nn/layer';
import { Value } from '../core/engine';

export type NeuronVisualizationData = {
  id: string;
  value: number;
  activation: number;
  gradient: number;
  weights: {
    connectionId: string;
    weight: number;
    gradient: number;
  }[];
};

export type LayerVisualizationData = {
  id: string;
  neurons: NeuronVisualizationData[];
};

export type NetworkVisualizationData = {
  layers: LayerVisualizationData[];
  inputs: {
    id: string;
    value: number;
  }[];
  loss?: number;
};

export class Visualizer {
  private model: MLP;
  
  constructor(model: MLP) {
    this.model = model;
  }
  
  getNetworkData(inputValues: number[]): NetworkVisualizationData {
    const inputs = inputValues.map(val => new Value(val));
    
    const output = this.model.call(inputs);
    
    const networkData: NetworkVisualizationData = {
      layers: [],
      inputs: inputValues.map((val, i) => ({
        id: `input-${i}`,
        value: val
      }))
    };
    
    const layers = (this.model as any).layers as Layer[];
    
    layers.forEach((layer, layerIndex) => {
      const layerData: LayerVisualizationData = {
        id: `layer-${layerIndex}`,
        neurons: []
      };
      
      const neurons = layer.getNeurons();
      neurons.forEach((neuron, neuronIndex) => {
        const neuronData: NeuronVisualizationData = {
          id: `neuron-${layerIndex}-${neuronIndex}`,
          value: 0,
          activation: 0,
          gradient: neuron.b.grad,
          weights: []
        };
        
        neuron.w.forEach((weight, weightIndex) => {
          const sourceId = layerIndex === 0 
            ? `input-${weightIndex}` 
            : `neuron-${layerIndex-1}-${weightIndex}`;
            
          neuronData.weights.push({
            connectionId: `${sourceId}-to-${neuronData.id}`,
            weight: weight.data,
            gradient: weight.grad
          });
        });
        
        layerData.neurons.push(neuronData);
      });
      
      networkData.layers.push(layerData);
    });
    
    return networkData;
  }
  
  visualizeForwardBackward(inputValues: number[], targetValue: number): NetworkVisualizationData & { loss: number } {
    const inputs = inputValues.map(val => new Value(val));
    
    this.model.zeroGrad();
    
    const predictions = this.model.call(inputs);
    
    const target = new Value(targetValue);
    const loss = predictions[0].sub(target).pow(2);
    
    loss.backward();
    
    const networkData = this.getNetworkData(inputValues);
    
    return {
      ...networkData,
      loss: loss.data
    };
  }
  
  getParameterSnapshot(): { name: string, value: number, gradient: number }[] {
    const params = this.model.parameters();
    return params.map((param, index) => ({
      name: `param_${index}`,
      value: param.data,
      gradient: param.grad
    }));
  }
  
  getGradientMagnitude(): number {
    const params = this.model.parameters();
    let sumSquaredGrads = 0;
    
    for (const param of params) {
      sumSquaredGrads += param.grad * param.grad;
    }
    
    return Math.sqrt(sumSquaredGrads);
  }
}