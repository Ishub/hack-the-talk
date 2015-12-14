import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
 
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
 import weka.classifiers.Evaluation;
 import java.util.Random;


public class KNN {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
 
	public static void main(String[] args) throws Exception {
		BufferedReader datafile = readDataFile("tdtest.arff");
 
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() -1);
 
		//do not use first and second
		Instance first = data.instance(0);
		//Instance second = data.instance(1);
		//data.delete(0);
		//data.delete(1);
 
		Classifier ibk = new IBk(3);
		int k=7;
		ibk.buildClassifier(data);
		

 Evaluation eval = new Evaluation(data);
 eval.evaluateModel(ibk, data);
 System.out.println(eval.toSummaryString("\nResults\n======\n", false));
  System.out.println(eval.toMatrixString());


	double class1 = ibk.classifyInstance(first);
		//double class2 = ibk.classifyInstance(second);
 

// Evaluation eval = new Evaluation(data);
 //eval.evaluateModel(cls, test);


		System.out.println("first: " + class1);
	}
}
