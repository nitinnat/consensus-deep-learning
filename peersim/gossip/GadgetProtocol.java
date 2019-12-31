/*
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
import org.nd4j.linalg.dataset.DataSet;
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

/**
 *  @author Nitin Nataraj
 */


public class GadgetProtocol implements CDProtocol {
	public static boolean flag = false;
	public static int t = 0;
	public static boolean optimizationDone = false;	
	public double EPSILON_VAL = 0.01;
	protected int lid;
	protected double lambda;
	protected int T;
	public static double[][] optimalB;
	public static int end = 0;
	public static boolean pushsumobserverflag = false;
	public static final int CONVERGENCE_COUNT = 10;
	private String protocol;
	private String resourcepath;


	/**
	 * Default constructor for configurable objects.
	 */
	public GadgetProtocol(String prefix) {
		lid = FastConfig.getLinkable(CommonState.getPid());
		
		protocol = Configuration.getString(prefix + "." + "prot", "pushsum1");
		
	}

	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverRequest(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}
	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverResponse(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}

	/**
	 * Clone an existing instance. The clone is considered 
	 * new, so it cannot participate in the aggregation protocol.
	 */
	public Object clone() {
		GadgetProtocol gp = null;
		try { gp = (GadgetProtocol)super.clone(); }
		catch( CloneNotSupportedException e ) {} // never happens
		return gp;
	}
	
	
	protected List<Node> getPeers(Node node) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) {
			List<Node> l = new ArrayList<Node>(linkable.degree());			
			for(int i=0;i<linkable.degree();i++) {
				l.add(linkable.getNeighbor(i));
			}
			return l;
		}
		else
			return null;						
	}			


public void nextCycle(Node node, int pid) {
		
		int iter = CDState.getCycle(); // Gets the current cycle of Gadget
		PegasosNode pn = (PegasosNode)node; // Initializes the Pegasos Node
		
		final String resourcepath = pn.resourcepath;
			System.out.println(iter);

			// If only 1 node exists - then we implement the centralized NN
			if (Network.size() == 1) {
				// Train NN for one epoch
				try {
					if(!pn.converged) {
						
						pn.neural_network.train(pn.train_features, pn.train_labels, 
												pn.test_features, pn.test_labels, 
												1, pn.csv_filename,(int)node.getID(), 1);
						
						// pn.train_loss represents the loss at the previous cycle
						// pn.neural_network.train_loss is the train_loss obtained after this cycle.
						double abs_change_in_train_loss = Math.abs(pn.neural_network.train_loss - pn.train_loss);
						pn.train_loss = pn.neural_network.train_loss;
						pn.test_loss = pn.neural_network.test_loss;
						
						if (abs_change_in_train_loss < pn.convergence_epsilon) {
							pn.num_converged_cycles += 1;
						}
						else {
							pn.num_converged_cycles = 0;
						}
						
						if (pn.num_converged_cycles >= pn.cycles_for_convergence) {
							pn.converged = true;
							System.out.println("Algorithm has converged for node " + pn.getID() + 
									" after " + pn.num_converged_cycles + " cycles. No further processing will be done.");
						}
					}
					
				} catch (IOException e) {
					e.printStackTrace();
				}
				
				
			}
	
			else {
				
				
				// check if node has converged
				if(!pn.converged) {
					
				
					// Select neighbor
					PegasosNode peer = (PegasosNode)selectNeighbor(node, pid);
					
					List<INDArray> pn_layer_outputs = pn.neural_network.feedforward(pn.train_features);
					List<INDArray> peer_layer_outputs = peer.neural_network.feedforward(peer.train_features);
					
					// Exchange loss vectors
					INDArray pn_output = pn_layer_outputs.get(pn_layer_outputs.size()-1);
					INDArray peer_output = peer_layer_outputs.get(peer_layer_outputs.size()-1);
					INDArray avg_output = pn_output.add(peer_output).mul(0.5);
					
					pn_layer_outputs.set(pn_layer_outputs.size()-1, avg_output);
					peer_layer_outputs.set(peer_layer_outputs.size()-1, avg_output);
					
					// Backprop on each node for all examples
					for(int j=0; j<pn.train_features.size(0);j++) {
						int [] rows = new int[] {j};
						INDArray cur_training_labels = pn.train_labels.getRows(rows);
						
						// Node pn
		    			INDArray pn_cur_training_data = pn.train_features.getRows(rows);
			    		INDArray pn_cur_first = pn_layer_outputs.get(0).getRows(rows); 
		    			INDArray pn_cur_second = pn_layer_outputs.get(1).getRows(rows);
		    			List<INDArray> pn_cur_layer_outputs = new ArrayList<INDArray>();
		    			pn_cur_layer_outputs.add(pn_cur_first);
		    			pn_cur_layer_outputs.add(pn_cur_second);
			    		pn.neural_network.backpropagate(pn_cur_layer_outputs, pn_cur_training_data, cur_training_labels);
			    		
			    		
			    		// Node peer
			    		// Backprop on peer only if it has not converged
			    		if (!peer.converged) {
			    			INDArray peer_cur_training_data = peer.train_features.getRows(rows);
				    		INDArray peer_cur_first = peer_layer_outputs.get(0).getRows(rows); 
			    			INDArray peer_cur_second = peer_layer_outputs.get(1).getRows(rows);
			    			List<INDArray> peer_cur_layer_outputs = new ArrayList<INDArray>();
			    			peer_cur_layer_outputs.add(peer_cur_first);
			    			peer_cur_layer_outputs.add(peer_cur_second);
				    		peer.neural_network.backpropagate(peer_cur_layer_outputs, peer_cur_training_data, cur_training_labels);
			    		}
			    		
		    		}
					
					// Determine algorithm convergence for node pn
					List<INDArray> train_outputs = pn.neural_network.feedforward(pn.train_features);
		        	INDArray train_preds = train_outputs.get(train_outputs.size()-1);
		        	List<Double> train_stats = NeuralNetwork.compute_stats(pn.train_labels, train_preds);
		        	double pn_train_loss = train_stats.get(0);
		        	double pn_abs_change_in_train_loss = Math.abs(pn_train_loss - pn.train_loss); 
		        	
		        	if (pn_abs_change_in_train_loss < pn.convergence_epsilon) {
						pn.num_converged_cycles += 1;
					}
		        	else {
						pn.num_converged_cycles = 0;
					}
		        	if (pn.num_converged_cycles >= pn.cycles_for_convergence) {
						pn.converged = true;
						System.out.println("Algorithm has converged for node " + pn.getID() + 
								" after " + pn.num_converged_cycles + " cycles. No further processing will be done.");
					}
					
		     
					
					
				}
				// After all the nodes have been processed in one cycle, we compute losses and accuracies
				if(pn.getID() == Network.size()-1) {
					
					
					String csv_filename = resourcepath + "/run" + pn.num_run + "/vpnn_results_temp_" + Network.size() + ".csv";
					String weights_filename = resourcepath + "/run" + pn.num_run + "/vpnn_weights_temp_" + Network.size() + ".csv";
					BufferedWriter bw;
					try {
						bw = new BufferedWriter(new FileWriter(csv_filename, true));
						for(int i=0; i < Network.size(); i++) {
				        	PegasosNode temp_node = (PegasosNode)Network.get(i);
				        	// Train Loss and Accuracy
				        	List<INDArray> temp_layer_outputs_train = temp_node.neural_network.feedforward(temp_node.train_features);
				        	INDArray temp_train_pred_outputs = temp_layer_outputs_train.get(temp_layer_outputs_train.size()-1);
				        	List<Double> train_stats = NeuralNetwork.compute_stats(temp_node.train_labels, temp_train_pred_outputs);
				        	double train_loss = train_stats.get(0);
				        	double train_acc = train_stats.get(1);
				        
				        	// Test Loss and Accuracy
				        	List<INDArray> temp_layer_outputs_test = temp_node.neural_network.feedforward(temp_node.test_features);
				        	INDArray temp_test_pred_outputs = temp_layer_outputs_test.get(temp_layer_outputs_test.size()-1);
				        	List<Double> test_stats = NeuralNetwork.compute_stats(temp_node.test_labels, temp_test_pred_outputs);
				        	double test_loss = test_stats.get(0);
				        	double test_acc = test_stats.get(1);
				        	
				        	// Train and Test AUC
				        	double train_auc = NeuralNetwork.compute_auc(temp_node.train_labels, temp_train_pred_outputs);
				        	double test_auc = NeuralNetwork.compute_auc(temp_node.test_labels, temp_test_pred_outputs);
				        	
				        	System.out.println("Iter: "+iter+" Node: "+ i + " TrainAcc: "+train_acc +" TestAcc: " + test_acc);
				        	System.out.println("TrainLoss: "+ train_loss + " TestLoss: "+ test_loss);
				        	System.out.println("TrainAUC: "+ train_auc + " TestAUC: "+ test_auc);
				        	bw.write(iter + "," + i + ","+ train_loss+ ","+test_loss+','+train_acc+","+test_acc+','+train_auc+","+test_auc);
							bw.write("\n");
						}
						bw.close();
					}
					catch (IOException e) {e.printStackTrace();}
					
			}	
		}
	}

	/**
	 * Selects a random neighbor from those stored in the {@link Linkable} protocol
	 * used by this protocol.
	 */
	protected Node selectNeighbor(Node node, int pid) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) 
			return linkable.getNeighbor(
					CommonState.r.nextInt(linkable.degree()));
		else
			return null;
	}

	public static void writeIntoFile(String millis) {
		File file = new File("exec-time.txt");
		 
		// if file doesnt exists, then create it
		if (!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		FileWriter fw;
		try {
			fw = new FileWriter(file.getAbsoluteFile(),true);

		BufferedWriter bw = new BufferedWriter(fw);
		bw.write(millis+"\n");
		bw.close();
		} catch (IOException e)
		
		 {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		

	}
	
	private INDArray crossEntropyLoss(INDArray predictions, INDArray labels) {

		/*
		 Computes the cross-entropy loss between predictions and labels array.
		 */
		int numRows = predictions.rows();
		int numCols = predictions.columns();
		
		INDArray batchLossVector = Nd4j.zeros(numRows, 1);
		for(int i=0;i<numRows;i++) {
			double loss = 0.0;
			for(int j=0;j<numCols;j++) {
				loss += ((labels.getDouble(i,j)) * Math.log(predictions.getDouble(i,j) + 1e-15)) + (
						(1-labels.getDouble(i,j)) * Math.log((1-predictions.getDouble(i,j)) + 1e-15)
						)
						;
				
			}
			batchLossVector.putScalar(i, 0, -loss);
		}
		return batchLossVector;
	}
}
