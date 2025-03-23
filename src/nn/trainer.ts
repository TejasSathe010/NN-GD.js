import { Value } from '../core/engine';
import { MLP } from './mlp';

type TrainingExample = {
  inputs: number[];
  target: number | number[];
};

type TrainingOptions = {
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: 'sgd' | 'momentum' | 'adam';
  momentumBeta?: number;
  adamBetas?: [number, number];
  adamEpsilon?: number;
  lossType?: 'mse' | 'bce';
  callback?: (epoch: number, loss: number, metrics: Record<string, number>) => void;
};

export class Trainer {
  private model: MLP;
  private options: TrainingOptions;
  private velocities?: number[];
  private m?: number[];
  private v?: number[];
  private t: number;

  constructor(model: MLP, options: Partial<TrainingOptions> = {}) {
    this.model = model;
    this.options = {
      learningRate: options.learningRate || 0.01,
      batchSize: options.batchSize || 1,
      epochs: options.epochs || 10,
      optimizer: options.optimizer || 'sgd',
      momentumBeta: options.momentumBeta || 0.9,
      adamBetas: options.adamBetas || [0.9, 0.999],
      adamEpsilon: options.adamEpsilon || 1e-8,
      lossType: options.lossType || 'mse',
      callback: options.callback
    };
    
    this.t = 0;
    
    // Initialize optimizer state if needed
    if (this.options.optimizer === 'momentum') {
      this.velocities = this.model.parameters().map(() => 0);
    } else if (this.options.optimizer === 'adam') {
      this.m = this.model.parameters().map(() => 0);
      this.v = this.model.parameters().map(() => 0);
    }
  }

  // Calculate loss based on loss type
  private calculateLoss(prediction: Value | Value[], target: Value | Value[]): Value {
    if (this.options.lossType === 'mse') {
      // Handle single value case
      if (!(prediction instanceof Array) && !(target instanceof Array)) {
        return prediction.sub(target).pow(2);
      }
      
      // Handle multi-output case
      if (prediction instanceof Array && target instanceof Array) {
        let totalLoss = new Value(0);
        for (let i = 0; i < prediction.length; i++) {
          totalLoss = totalLoss.add(prediction[i].sub(target[i]).pow(2));
        }
        return totalLoss;
      }
    } 
    else if (this.options.lossType === 'bce') {
      // Binary cross entropy
      if (!(prediction instanceof Array) && !(target instanceof Array)) {
        // Clamp prediction to avoid log(0)
        const p = prediction.data > 0.9999 ? new Value(0.9999) : 
                 prediction.data < 0.0001 ? new Value(0.0001) : prediction;
        
        // BCE = -t*log(p) - (1-t)*log(1-p)
        const term1 = target.mul(new Value(Math.log(p.data))).mul(new Value(-1));
        const term2 = new Value(1).sub(target).mul(new Value(Math.log(1 - p.data))).mul(new Value(-1));
        return term1.add(term2);
      }
    }
    
    // Default fallback to MSE
    return prediction instanceof Array ? 
      prediction[0].sub(target instanceof Array ? target[0] : target).pow(2) :
      prediction.sub(target instanceof Array ? target[0] : target).pow(2);
  }

  // Update parameters based on chosen optimizer
  private updateParameters(parameters: Value[]): void {
    if (this.options.optimizer === 'sgd') {
      // Simple SGD
      parameters.forEach(param => {
        param.data -= this.options.learningRate * param.grad;
      });
    } 
    else if (this.options.optimizer === 'momentum' && this.velocities) {
      // SGD with momentum
      parameters.forEach((param, i) => {
        this.velocities![i] = this.options.momentumBeta! * this.velocities![i] - 
                             this.options.learningRate * param.grad;
        param.data += this.velocities![i];
      });
    } 
    else if (this.options.optimizer === 'adam' && this.m && this.v) {
      // Adam optimizer
      this.t += 1;
      const [beta1, beta2] = this.options.adamBetas!;
      const epsilon = this.options.adamEpsilon!;
      
      parameters.forEach((param, i) => {
        // Update biased first moment estimate
        this.m![i] = beta1 * this.m![i] + (1 - beta1) * param.grad;
        // Update biased second raw moment estimate
        this.v![i] = beta2 * this.v![i] + (1 - beta2) * param.grad * param.grad;
        
        // Bias-corrected estimates
        const mCorrected = this.m![i] / (1 - Math.pow(beta1, this.t));
        const vCorrected = this.v![i] / (1 - Math.pow(beta2, this.t));
        
        // Update parameters
        param.data -= this.options.learningRate * mCorrected / (Math.sqrt(vCorrected) + epsilon);
      });
    }
  }

  // Main training loop
  train(dataset: TrainingExample[]): {losses: number[], metrics: any[]} {
    const losses: number[] = [];
    const metrics: any[] = [];
    
    for (let epoch = 0; epoch < this.options.epochs; epoch++) {
      let epochLoss = 0;
      
      // Shuffle dataset for stochastic training
      const shuffledData = [...dataset].sort(() => Math.random() - 0.5);
      
      // Process in batches
      for (let i = 0; i < shuffledData.length; i += this.options.batchSize) {
        const batch = shuffledData.slice(i, i + this.options.batchSize);
        let batchLoss = 0;
        
        // Zero gradients before batch
        this.model.zeroGrad();
        
        // Process each example in the batch
        for (const example of batch) {
          // Convert inputs to Value objects
          const inputs = example.inputs.map(x => new Value(x));
          
          // Forward pass
          const predictions = this.model.call(inputs);
          
          // Convert target to Value object(s)
          const targets = Array.isArray(example.target) 
            ? example.target.map(t => new Value(t))
            : new Value(example.target);
          
          // Calculate loss
          const loss = this.calculateLoss(predictions, targets);
          batchLoss += loss.data;
          
          // Backward pass
          loss.backward();
        }
        
        // Average gradients over batch
        if (batch.length > 1) {
          const parameters = this.model.parameters();
          parameters.forEach(param => {
            param.grad /= batch.length;
          });
        }
        
        // Update parameters
        this.updateParameters(this.model.parameters());
        
        // Update epoch loss
        epochLoss += batchLoss / batch.length;
      }
      
      // Average loss over all batches
      epochLoss /= Math.ceil(dataset.length / this.options.batchSize);
      losses.push(epochLoss);
      
      // Calculate metrics on the entire dataset
      const currentMetrics = this.calculateMetrics(dataset);
      metrics.push(currentMetrics);
      
      // Call callback if provided
      if (this.options.callback) {
        this.options.callback(epoch, epochLoss, currentMetrics);
      }
    }
    
    return { losses, metrics };
  }
  
  // Calculate metrics like accuracy
  private calculateMetrics(dataset: TrainingExample[]): Record<string, number> {
    // For binary classification
    if (this.options.lossType === 'bce') {
      let correct = 0;
      
      for (const example of dataset) {
        const inputs = example.inputs.map(x => new Value(x));
        const predictions = this.model.call(inputs);
        const prediction = predictions[0].data;
        const target = Array.isArray(example.target) ? example.target[0] : example.target;
        
        if ((prediction >= 0.5 && target >= 0.5) || (prediction < 0.5 && target < 0.5)) {
          correct++;
        }
      }
      
      return { accuracy: correct / dataset.length };
    }
    
    // For regression, return mean absolute error
    let totalError = 0;
    
    for (const example of dataset) {
      const inputs = example.inputs.map(x => new Value(x));
      const predictions = this.model.call(inputs);
      const prediction = predictions[0].data;
      const target = Array.isArray(example.target) ? example.target[0] : example.target;
      
      totalError += Math.abs(prediction - target);
    }
    
    return { mae: totalError / dataset.length };
  }
}