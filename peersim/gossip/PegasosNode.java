/*
 * Peersim-Gadget : A Gadget protocol implementation in peersim based on the paper
 * Chase Henzel, Haimonti Dutta
 * GADGET SVM: A Gossip-bAseD sub-GradiEnT SVM Solver   
 * 
 * Copyright (C) 2012
 * Deepak Nayak 
 * Columbia University, Computer Science MS'13
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 */
package peersim.gossip;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.File;
import java.io.IOException;
import java.io.LineNumberReader;
import java.net.MalformedURLException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.learning.config.Adam;
import peersim.config.*;
import peersim.core.*;

/**
 * Class PegasosNode
 * An implementation of {@link Node} which can handle external resources. 
 * It is based on {@link GeneralNode} and extended it to be used with pegasos solver.
 * At the start of node, each node has some associated data file, on which it calls
 * training function through a jni call. 
 * It will take the resourcepath from config file.
 * This is another implementation of {@link Node} class that is used to compose the
 * p2p {@link Network}, where each node can handle an external resource.
 * @author Nitin Nataraj
 */
public class PegasosNode implements Node {

	// ================= fields ========================================
	// =================================================================

	
	// Neural net params in the config file
	private static final String PAR_LEARNING_RATE = "learningrate";
	private static final String PAR_BATCH_SIZE = "batchsize";
	private static final String PAR_INITIALIZATION = "initmethod";
	private static final String PAR_ACTIVATION_METHOD = "activation";
	private static final String PAR_EPOCHS = "epochs";
	private static final String PAR_TRAINLEN = "trainlen";
	private static final String PAR_TESTLEN = "testlen";
	private static final String PAR_NUMCLASSES = "numclasses";
	private static final String PAR_PATH = "resourcepath";	
	private static final String PAR_SIZE = "size";
	private static final String PAR_NUMHIDDEN = "numhidden";
	private static long counterID = -1; // used to generate unique IDs 
	protected Protocol[] protocol = null; //The protocols on this node.
	
 	//learning parameters of the Neural Network 
    private int train_length, test_length;
	public double learning_rate;
	public int epochs;
	public int batch_size;
	public int initmethod; // Method to initialize the weights of the NN. 0 - Random, 1 - Xavier
	public int activationmethod; // Activation function of layers of NN. 0 - Identity, 1 - Sigmoid, 2 - RELU, 3 - Tanh 
	/**
	 * The current index of this node in the node
	 * list of the {@link Network}. It can change any time.
	 * This is necessary to allow
	 * the implementation of efficient graph algorithms.
	 */
	private int index;

	/**
	 * The fail state of the node.
	 */
	protected int failstate = Fallible.OK;

	/**
	 * The ID of the node. It should be final, however it can't be final because
	 * clone must be able to set it.
	 */
	private long ID;

	/**
	 * The prefix for the resources file. All the resources file will be in prefix 
	 * directory. later it should be taken from configuration file.
	 */
	public String resourcepath;
	private int num_nodes;
	public int num_run;
	public boolean converged = false;
	public long num_features;
	public DataSet train_set;
	public DataSet test_set;
	public INDArray train_features, test_features, train_labels, test_labels;
	public int num_classes;
	public int num_hidden_nodes;
	public NeuralNetwork neural_network;
	
	// ================ constructor and initialization =================
	// =================================================================
	/** Used to construct the prototype node. This class currently does not
	 * have specific configuration parameters and so the parameter
	 * <code>prefix</code> is not used. It reads the protocol components
	 * (components that have type {@value peersim.core.Node#PAR_PROT}) from
	 * the configuration.
	 */
	
	private static DataSet readCSVDataset(
	        String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
	        throws IOException, InterruptedException {

	        RecordReader rr = new CSVRecordReader();
	        rr.initialize(new FileSplit(new File(csvFileClasspath)));
	        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
	        return iterator.next();
	    }
	
	public PegasosNode(String prefix) {
		System.out.println(prefix);
		String[] names = Configuration.getNames(PAR_PROT);
		resourcepath = (String)Configuration.getString(prefix + "." + PAR_PATH);
		learning_rate = Configuration.getDouble(prefix + "." + PAR_LEARNING_RATE, 1e-6);
		batch_size = Configuration.getInt(prefix + "." + PAR_BATCH_SIZE, 4);
		epochs = Configuration.getInt(prefix + "." + PAR_EPOCHS, 1);
		initmethod = Configuration.getInt(prefix + "." + PAR_INITIALIZATION, 1);
		activationmethod = Configuration.getInt(prefix + "." + PAR_ACTIVATION_METHOD, 0);
		train_length = Configuration.getInt(prefix + "." + PAR_TRAINLEN, 0);
		test_length = Configuration.getInt(prefix + "." + PAR_TESTLEN, 0);
		num_classes = Configuration.getInt(prefix + "." + PAR_NUMCLASSES, 1);
		num_hidden_nodes = Configuration.getInt(prefix + "." + PAR_NUMHIDDEN, 1);
		
		System.out.println("model file and train file are saved in: " + resourcepath);
		CommonState.setNode(this);
		ID = nextID();
		protocol = new Protocol[names.length];
		for (int i=0; i < names.length; i++) {
			CommonState.setPid(i);
			Protocol p = (Protocol) 
					Configuration.getInstance(names[i]);
			protocol[i] = p; 
		}
		num_nodes = Configuration.getInt(prefix + "." + PAR_SIZE, 20);
		num_run = Configuration.getInt(prefix + "." + "run", 1);
		System.out.println("Number of nodes is ####### "+num_nodes);
	}
	
	/**
	 * Used to create actual Node by calling clone() on a prototype node. So, actually 
	 * a Node constructor is only called once to create a prototype node and after that
	 * all nodes are created by cloning it.
	 
	 */
	
	public Object clone() {
		
		
		PegasosNode result = null;
		
		try { 
			result=(PegasosNode)super.clone(); 
			}
		catch( CloneNotSupportedException e ) {} // never happens
		
		result.protocol = new Protocol[protocol.length];
		CommonState.setNode(result);
		result.ID = nextID();
		for(int i=0; i<protocol.length; ++i) {
			CommonState.setPid(i);
			result.protocol[i] = (Protocol)protocol[i].clone();
		}
		System.out.println("creating node with ID: " + result.getID());
		
		// Determine base dataset name
		String[] temp_data = resourcepath.split("/");
		String base_dataset_name = temp_data[temp_data.length - 1];
        
		// Get train file and test file paths
        String localTrainFilepath = "";
        String localTestFilepath = "";
        
        if (Network.size() == 1) {
        	localTrainFilepath = resourcepath + "/" + base_dataset_name + "_train_binary.csv";
        	localTestFilepath = resourcepath + "/" + base_dataset_name + "_test_binary.csv";
        }
        
        else {
        	localTrainFilepath = resourcepath + "/" + base_dataset_name + "_train_" + result.getID() + ".csv";
        	localTestFilepath = resourcepath + "/" + base_dataset_name + "_test_" + result.getID() + ".csv";
        }
        
        
		// Create a folder for this run if it does not exist
		File directory = new File(resourcepath + "/run" + result.num_run);
	    if (! directory.exists()){
	        directory.mkdir();
	    
	    }
	    
	    // If this is the first node in the cycle, then create the results file here.
	    if (result.getID() == 0) {
	    	// Create headers to store the results
	    	System.out.println("Creating csv file to store the results.");
	    	String csv_filename = resourcepath + "/run" + result.num_run + "/vpnn_results_temp_" + Network.size() + ".csv";
			System.out.println("Storing in " + csv_filename);
			String opString = "Iter,Node,TrainLoss,TestLoss";
			try {
				BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename));
				bw.write(opString);
				bw.write("\n");
				bw.close();
			}
			catch(Exception e) {}
	    }
		
		
		try{
		    // Load training data    
			System.out.println("Reading train data from: " + localTrainFilepath);
		    result.train_set = readCSVDataset(
		           localTrainFilepath,
		            train_length, 0, 1
		        );
		    // Load test data
		    System.out.println("Reading test data from: " + localTestFilepath);
		    result.test_set = readCSVDataset(
		    		localTestFilepath,
		    		test_length, 0, 1
			        );
		    result.train_features = result.train_set.getFeatures();
		    result.train_labels = result.train_set.getLabels();
		    result.test_features = result.test_set.getFeatures();
		    result.test_labels = result.test_set.getLabels();
		    
		    
		    // Add bias to both train and test features
		    result.train_features = NeuralNetwork.add_bias_to_input(result.train_features);
		    result.test_features = NeuralNetwork.add_bias_to_input(result.test_features);
		    
		    System.out.println(result.train_features.shapeInfoToString());
		    System.out.println(result.test_features.shapeInfoToString());
	        int num_features = (int)result.train_features.size(1);
	        int num_outputs = 1;
		        
	        // Create this Node's Neural Network instance
	        NeuronLayer layer1 = new NeuronLayer(num_features, num_hidden_nodes);
	        NeuronLayer layer2 = new NeuronLayer(num_hidden_nodes, num_outputs);
	        
	        // Combine the layers to create a neural network
	        result.neural_network = new NeuralNetwork(layer1, layer2);
	        
	        System.out.println("Initial weights \n");
	        result.neural_network.print_weights();

		} catch (Exception e) {e.printStackTrace();}
		return result;
		

	}

 
	/** returns the next unique ID */
	private long nextID() {

		return counterID++;
	}

	// =============== public methods ==================================
	// =================================================================


	public void setFailState(int failState) {

		// after a node is dead, all operations on it are errors by definition
		if(failstate==DEAD && failState!=DEAD) throw new IllegalStateException(
				"Cannot change fail state: node is already DEAD");
		switch(failState)
		{
		case OK:
			failstate=OK;
			break;
		case DEAD:
			//protocol = null;
			index = -1;
			failstate = DEAD;
			for(int i=0;i<protocol.length;++i)
				if(protocol[i] instanceof Cleanable)
					((Cleanable)protocol[i]).onKill();
			break;
		case DOWN:
			failstate = DOWN;
			break;
		default:
			throw new IllegalArgumentException(
					"failState="+failState);
		}
	}

	public int getFailState() { return failstate; }

	public boolean isUp() { return failstate==OK; }

	public Protocol getProtocol(int i) { return protocol[i]; }

	public int protocolSize() { return protocol.length; }

	public int getIndex() { return index; }

	public void setIndex(int index) { this.index = index; }
        
        
	/**
	 * Returns the ID of this node. The IDs are generated using a counter
	 * (i.e. they are not random).
	 */
	public long getID() { return ID; }

	public String toString() 
	{
		StringBuffer buffer = new StringBuffer();
		buffer.append("ID: "+ID+" index: "+index+"\n");
		for(int i=0; i<protocol.length; ++i)
		{
			buffer.append("protocol[" + i +"]=" + protocol[i] + "\n");
		}
		return buffer.toString();
	}

	/** Implemented as <code>(int)getID()</code>. */
	public int hashCode() { return (int)getID(); }

	

}
