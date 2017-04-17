package analysis;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map.Entry;

import util.EvalUtil;

import kdtree.KDTree;
import data.RandomData;

public class KDTreeAnalysis {
	
	public static void main(String[] args) throws Exception {
		// TODO - run tests here
	}
	
	/** Tests the query time and distance for a random data set and test set
	 * @param n - the number of points of the dataset
	 * @param D - the dimension of the data points
	 * @param nTest - the number of points to test
	 * @param alphas - a set of alphas to test
	 * @return - an object of class Result, which has the average time and distance for a single query
	 */
	public static Collection<TestResult> TestKDTree(Integer n, Integer D, Integer nTest, double[] alphas) {
		HashMap<Integer, Double[]> documents = RandomData.RandomDataset(n, D);
		HashMap<Integer, Double[]> testPoints = RandomData.RandomDataset(nTest, D);
		
		double[] key = new double[D];
		KDTree randTree = new KDTree(D);
		for (Entry<Integer, Double[]> document : documents.entrySet()) {
			for (int i = 0; i < D; i++) {
				key[i] = document.getValue()[i];
			}
			randTree.insert(key, document.getKey());
		}
		ArrayList<TestResult> times = new ArrayList<TestResult>();
		for (int a = 0; a < alphas.length; a++) {
			long startTime = System.nanoTime();
			double cumDist = 0.0;
			for (Double[] testPt : testPoints.values()) {
				for (int i = 0; i < D; i++) {
					key[i] = testPt[i];
				}
				Integer docId = (Integer)randTree.nearest(key, alphas[a]);
				cumDist += EvalUtil.Distance(testPt, documents.get(docId));
			}
			times.add(new TestResult("KDTree", n, D, alphas[a],
					((System.nanoTime() - startTime) / Math.pow(10, 9)) / (nTest + 0.0),
					cumDist / (nTest + 0.0)));
		}
		return times;
	}
}
