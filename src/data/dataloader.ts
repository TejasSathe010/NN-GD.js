type DatasetExample = {
    inputs: number[];
    target: number | number[];
  };
  
  export class DataLoader {
    private data: DatasetExample[];
    private batchSize: number;
    private shuffle: boolean;
    
    constructor(data: DatasetExample[], batchSize = 32, shuffle = true) {
      this.data = data;
      this.batchSize = batchSize;
      this.shuffle = shuffle;
    }
    
    get numBatches(): number {
      return Math.ceil(this.data.length / this.batchSize);
    }
    
    get size(): number {
      return this.data.length;
    }
    
    *getBatches(): Generator<DatasetExample[]> {
      const data = [...this.data];
      
      if (this.shuffle) {
        for (let i = data.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [data[i], data[j]] = [data[j], data[i]];
        }
      }
      
      for (let i = 0; i < data.length; i += this.batchSize) {
        yield data.slice(i, i + this.batchSize);
      }
    }
    
    static splitData(data: DatasetExample[], trainRatio = 0.8): [DatasetExample[], DatasetExample[]] {
      const shuffled = [...data].sort(() => Math.random() - 0.5);
      
      const splitIndex = Math.floor(shuffled.length * trainRatio);
      const trainData = shuffled.slice(0, splitIndex);
      const valData = shuffled.slice(splitIndex);
      
      return [trainData, valData];
    }
    
    static createXORDataset(numExamples = 100): DatasetExample[] {
      const dataset: DatasetExample[] = [];
      
      for (let i = 0; i < numExamples; i++) {
        const x1 = Math.random() > 0.5 ? 1 : 0;
        const x2 = Math.random() > 0.5 ? 1 : 0;
        const target = x1 !== x2 ? 1 : 0; // XOR logic
        
        dataset.push({
          inputs: [x1, x2],
          target
        });
      }
      
      return dataset;
    }
    
    static createRegressionDataset(numExamples = 100, noise = 0.1): DatasetExample[] {
      const dataset: DatasetExample[] = [];
      
      for (let i = 0; i < numExamples; i++) {
        const x = Math.random() * 10 - 5;
        const noise_val = (Math.random() - 0.5) * 2 * noise;
        const y = x * x + 2 * x + 1 + noise_val;
        
        dataset.push({
          inputs: [x],
          target: y
        });
      }
      
      return dataset;
    }
  }