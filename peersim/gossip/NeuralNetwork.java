package peersim.gossip;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;


import org.deeplearning4j.eval.Evaluation;
import org.apache.commons.math3.optim.nonlinear.vector.Weight;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Dot;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.eval.ROCBinary;

import java.util.Arrays;
import java.lang.Integer;

import java.io.FileReader;

import java.io.LineNumberReader;
import peersim.gossip.PegasosNode;

import peersim.config.Configuration;
import peersim.config.FastConfig;
import peersim.core.*;
import peersim.cdsim.*;


import java.net.MalformedURLException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.text.ParseException;
import java.io.BufferedReader;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

public class NeuralNetwork {
	NeuronLayer layer1;
	NeuronLayer layer2;
	double learning_rate;
	double train_loss;
	double test_loss;
	double train_auc;
	double test_auc;
	
	public static DataSet readCSVDataset(
	        String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
	        throws IOException, InterruptedException {

	        RecordReader rr = new CSVRecordReader();
	        rr.initialize(new FileSplit(new File(csvFileClasspath)));
	        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
	        return iterator.next();
	    }
	public NeuralNetwork(NeuronLayer layerA, NeuronLayer layerB, double lr) {
		layer1 = layerA;
		layer2 = layerB;
		learning_rate = lr;
		train_loss = -1;
		test_loss = -1;
		train_auc = -1;
		test_auc = -1;
	}
	
	public INDArray _sigmoid(INDArray x) {	
		return sigmoid(x);
	}
	
	public INDArray _sigmoid_derivative(INDArray x) {
		// x(1-x) = x - x^2
		return x.sub(x.mul(x).mul(-1));
	}
	
	public INDArray _tanh(INDArray x) {
		return tanh(x);
	}
	
	public INDArray _tanh_derivative(INDArray x) {
		return (tanh(x).mul(tanh(x)).sub(1)).mul(-1);
		
	}
	
	public INDArray _relu(INDArray x) {
		return max(x,  0);
	}
	
	public INDArray _relu_derivative(INDArray x) {
		return x.cond(Conditions.greaterThan(0));
	}
    
	public List<INDArray> feedforward(INDArray inputs) {
		INDArray output_from_layer1 = _sigmoid(inputs.mmul(layer1.synaptic_weights));
		INDArray output_from_layer2 = _sigmoid(output_from_layer1.mmul(layer2.synaptic_weights));
		List<INDArray> L = new ArrayList<INDArray>(2);
		L.add(output_from_layer1);
		L.add(output_from_layer2);
		return L;
	}

	public void backpropagate(List<INDArray> layer_outputs, INDArray cur_training_data, INDArray cur_training_labels) {
		
		INDArray output_from_layer_1 = layer_outputs.get(0);
		INDArray output_from_layer_2 = layer_outputs.get(1);
		
		INDArray layer2_error = cur_training_labels.sub(output_from_layer_2);
	    INDArray layer2_delta = layer2_error.mul(_sigmoid_derivative(output_from_layer_2));
	    
	    INDArray layer1_error = layer2_delta.mmul(layer2.synaptic_weights.transpose());
	    INDArray layer1_delta = layer1_error.mul(_sigmoid_derivative(output_from_layer_1));
	    
	    INDArray layer1_adjustment = cur_training_data.transpose().mmul(layer1_delta).mul(learning_rate);
	    INDArray layer2_adjustment = output_from_layer_1.transpose().mmul(layer2_delta).mul(learning_rate);
	    
	    layer1.synaptic_weights.addi(layer1_adjustment);
	    layer2.synaptic_weights.addi(layer2_adjustment);
	}
	
    public void print_weights() {
		System.out.println("Layer 1 Weights: " + layer1.synaptic_weights);
		System.out.println("Layer 2 Weights: " + layer2.synaptic_weights);
	}
	
    public static List<Double> compute_stats(INDArray true_outputs, INDArray pred_outputs) {
		INDArray layer2_error = true_outputs.sub(pred_outputs);
	    double loss = layer2_error.mul(layer2_error).mul(0.5).sum(0).getDouble(0) / true_outputs.size(0);
	    double acc = accuracy(true_outputs, pred_outputs);
	    List<Double> res = new ArrayList<Double>();
	    res.add(loss);
	    res.add(acc);
	    return res;
    	
    }
    
    public static void normalizeMatrix(INDArray toNormalize) {
        INDArray columnMeans = toNormalize.mean(0);
        toNormalize.subiRowVector(columnMeans);
        INDArray std = toNormalize.std(0);
        std.addi(Nd4j.scalar(1e-12));
        toNormalize.diviRowVector(std);
    }
    
    public static double accuracy(INDArray y_true, INDArray y_pred) {
    	final Evaluation evaluation = new Evaluation(0.5);
    	evaluation.eval(y_true, y_pred);
    	//System.out.println(evaluation.stats());
    	return evaluation.accuracy();
        
    }
    
    public static double compute_auc(INDArray y_true, INDArray y_pred) {
    	ROCBinary roc = new ROCBinary(0);
        roc.eval(y_true, y_pred);
        double auc = roc.calculateAUC(0);
        return auc;
        	
    }
    
    public String weights_to_string() {
    	String result = "";
    	int numRows =layer1.synaptic_weights.rows();
		int numCols = layer1.synaptic_weights.columns();
		
    	for(int i=0;i<numRows;i++) {
			for(int j=0;j<numCols;j++) {
				result = result + layer1.synaptic_weights.getDouble(i,j) + ",";
			}
    	
    	}
    	
    	// Removing trailing comma
    	result = result.substring(0, result.length() - 2);
    	
    	numRows =layer2.synaptic_weights.rows();
		numCols = layer2.synaptic_weights.columns();
		result = result + "\n";
    	for(int i=0;i<numRows;i++) {
			for(int j=0;j<numCols;j++) {
				result = result + layer2.synaptic_weights.getDouble(i,j) + ",";
			}
    	
    	}
    	
    	// Removing trailing comma
    	result = result.substring(0, result.length() - 2);
    	
    	
    	return result;
    }
    
    public static void scaleMinMax(double min, double max, INDArray toScale) {
        //X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) X_scaled = X_std * (max - min) + min

        INDArray min2 = toScale.min(0);
        INDArray max2 = toScale.max(0);

        INDArray std = toScale.subRowVector(min2).diviRowVector(max2.sub(min2));

        INDArray scaled = std.mul(max - min).addi(min);
        toScale.assign(scaled);
    }
    
    public static INDArray add_bias_to_input(INDArray x) {
    	// Adds another input column with all ones as the first column of the input matrix
    	//INDArray y = Nd4j.append(x, 1, 1, 1);
    	//return y;
    	INDArray bias = Nd4j.ones(x.size(0), 1);
    	INDArray arrayWithBias = Nd4j.concat(1, bias, x);
    	return arrayWithBias;
    }
    
    
    
    public void train(INDArray training_set_inputs, INDArray training_set_outputs,
    		INDArray testing_set_inputs, INDArray testing_set_outputs,
    		int number_of_training_iterations, String csv_filename, int node, int mode, boolean converged) throws IOException {
    	BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename, true));
    		
    		
	    	for (int i=0; i<number_of_training_iterations;i++) {
	    		
	    		if(!converged) {
	    		for(int j=0; j<training_set_inputs.size(0);j++) {
	    			// Feedforward
	    			int [] rows = new int[] {j};
	    			INDArray cur_training_data = training_set_inputs.getRows(rows);
	    			INDArray cur_training_labels = training_set_outputs.getRows(rows);
		    		List<INDArray> layer_outputs = feedforward(cur_training_data);
		    		
		    		// Backprop
		    		backpropagate(layer_outputs, cur_training_data, cur_training_labels);
	    		
	    			}
	    		}
	    		// compute train loss
	    		List<INDArray> train_layer_outputs = feedforward(training_set_inputs);
	    	    List<Double> train_res = compute_stats(training_set_outputs, train_layer_outputs.get(train_layer_outputs.size()-1));
	    	    
	    	    // compute test loss
	    	    List<INDArray> test_layer_outputs = feedforward(testing_set_inputs);
	    	    List<Double> test_res = compute_stats(testing_set_outputs, test_layer_outputs.get(test_layer_outputs.size()-1));
	    	    
	    	    double train_loss = train_res.get(0);
	    	    double train_acc = train_res.get(1);
	    	    double test_loss = test_res.get(0);
	    	    double test_acc = test_res.get(1);
	    	    
	    	    INDArray train_pred = train_layer_outputs.get(train_layer_outputs.size()-1);//.cond(Conditions.greaterThan(0.5));
	    	    INDArray test_pred = test_layer_outputs.get(test_layer_outputs.size()-1);//.cond(Conditions.greaterThan(0.5));
	    	    double train_auc = compute_auc(training_set_outputs, train_pred);
	    		double test_auc = compute_auc(testing_set_outputs, test_pred);
	    	    
	    		// Update class members
	    		this.train_loss = train_loss;
	    		this.test_loss = test_loss;
	    		this.train_auc = train_auc;
	    		this.test_auc = test_auc;
	    		
	    		// Write into output files
	    		int iter;
	    		if(mode == 0) {
		    		iter = i;
		    	}
		    	else {
		    		iter = CDState.getCycle();
		    	}
		    	   
	    		System.out.println("iter" + iter + " train_loss: " + train_loss +" test_loss: "+ test_loss +
	    				" train_acc: " + train_acc + " test_acc: " + test_acc);
	    		System.out.println("Train AUC: " + train_auc + " Test AUC: " + test_auc);
	    		
				// Write to file
				bw.write(iter + "," + node + ","+ train_loss+ ","+test_loss+','+train_acc+","+test_acc+","+train_auc+","+test_auc
						+","+converged+","+0);
				bw.write("\n");
				
	    	}
	    	bw.close();
    }
	
    public void write_weights_into_file(String weights_filename) {
    	File f = new File(weights_filename);
		Nd4j.writeTxt(layer1.synaptic_weights.getColumn(0), weights_filename);
    	System.out.println(Arrays.toString(layer1.synaptic_weights.getColumn(0).shape()));
    }
    
    public void train_on_all(INDArray training_set_inputs, INDArray training_set_outputs,
    		INDArray testing_set_inputs, INDArray testing_set_outputs,
    		int number_of_training_iterations, String csv_filename) throws IOException {
    	INDArray output_from_layer_1, output_from_layer_2;
    	String opString = "Iter,TrainLoss,TestLoss";
    	
		BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename));
		bw.write(opString);
		bw.write("\n");
    	int iter;
	    	for (int i=0; i<number_of_training_iterations;i++) {
	    		// Feedforward
	    			
	    			INDArray cur_training_data = training_set_inputs;
	    			INDArray cur_training_labels = training_set_outputs;
		    		List<INDArray> layer_outputs = feedforward(cur_training_data);
		    		
		    		output_from_layer_1 = layer_outputs.get(0);
		    		output_from_layer_2 = layer_outputs.get(1);
		    		
		    		INDArray layer2_error = cur_training_labels.sub(output_from_layer_2);
		    	    INDArray layer2_delta = layer2_error.mul(_sigmoid_derivative(output_from_layer_2));
		    	    
		    	    INDArray layer1_error = layer2_delta.mmul(layer2.synaptic_weights.transpose());
		    	    INDArray layer1_delta = layer1_error.mul(_sigmoid_derivative(output_from_layer_1));
		    	    
		    	    INDArray layer1_adjustment = cur_training_data.transpose().mmul(layer1_delta);
		    	    INDArray layer2_adjustment = output_from_layer_1.transpose().mmul(layer2_delta);
		    	    
		    	    layer1.synaptic_weights.addi(layer1_adjustment).mul(0.01);
		    	    layer2.synaptic_weights.addi(layer2_adjustment).mul(0.01);
	    		
	    		
		    	 
	    		// compute train loss
	    		//List<INDArray> layer_outputs = feedforward(training_set_inputs);
	    		//output_from_layer_1 = layer_outputs.get(0);
	    		//output_from_layer_2 = layer_outputs.get(1);
	    		//INDArray layer2_error = training_set_outputs.sub(output_from_layer_2);
	    	    double train_loss = layer2_error.mul(layer2_error).mul(0.5).sum(0).getDouble(0);
	    	    
	    	    
	    	    // compute test loss
	    	    List<INDArray> test_layer_outputs = feedforward(testing_set_inputs);
	    		INDArray test_output_from_layer_2 = test_layer_outputs.get(1);
	    		INDArray test_layer2_error = testing_set_outputs.sub(test_output_from_layer_2);
	    		double test_loss = test_layer2_error.mul(test_layer2_error).mul(0.5).sum(0).getDouble(0);
	    	    
	    		System.out.println("iter" + i + " train_loss: " + train_loss +" test_loss: "+ test_loss);
	    		
				bw.write(i + "," + train_loss+ ","+test_loss);
				bw.write("\n");
	    	}
	    	
    	bw.close();
    }
    
    public static void main(String args[]) throws IOException, InterruptedException {
        // Create layer 1 (4 neurons, each with 3 inputs)
        NeuronLayer layer1 = new NeuronLayer(501, 50, 12345);
        NeuronLayer layer2 = new NeuronLayer(50, 1, 12345);
        double learning_rate = 0.01;
        
        // Combine the layers to create a neural network
        NeuralNetwork neural_network = new NeuralNetwork(layer1, layer2, learning_rate);
        
        System.out.println("Initial weights \n");
        neural_network.print_weights();

        
        String localTrainFilepath = "C:/Users/Nitin/Documents/consensus-dl/data/madelon/madelon_train_binary.csv";
    	String localTestFilepath = "C:\\Users\\Nitin\\Documents\\consensus-dl\\data\\madelon\\madelon_test_binary.csv";
        
    	DataSet trainSet = readCSVDataset(
		           localTrainFilepath,
		            4800, 0, 1
		        );
		    
	    DataSet testSet = readCSVDataset(
	    		localTestFilepath,
	    		1200, 0, 1
		        );
	    
	    // Get train and test features and labels
	    INDArray training_set_inputs = trainSet.getFeatures();
	    INDArray training_set_outputs = trainSet.getLabels();
	    INDArray testing_set_inputs = testSet.getFeatures();
	    INDArray testing_set_outputs = testSet.getLabels();
	    
	    
	    // normalize input in place
	    // scaleMinMax(-1, 1, training_set_inputs);
	    // scaleMinMax(-1, 1, testing_set_inputs);
	    
	    System.out.println("Min " + training_set_inputs.minNumber());
	    System.out.println("Min " + training_set_inputs.maxNumber());
	    // add bias
	    training_set_inputs = add_bias_to_input(training_set_inputs);
	    testing_set_inputs = add_bias_to_input(testing_set_inputs);
	    
	    
	    System.out.println("Creating csv file to store the results.");
    	String csv_filename = "C:\\Users\\Nitin\\Documents\\consensus-dl\\data\\something.csv";
		System.out.println("Storing in " + csv_filename);
		
		
        //System.out.println(training_set_inputs);
		String opString = "Iter,Node,TrainLoss,TestLoss,TrainAccuracy,TestAccuracy";
    	
		BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename));
		bw.write(opString);
		bw.write("\n");
        //neural_network.train(training_set_inputs, training_set_outputs, testing_set_inputs, testing_set_outputs, 200, csv_filename, 0, 0);
		bw.close();
        
    }
}
