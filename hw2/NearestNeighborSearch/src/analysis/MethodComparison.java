package analysis;

import java.util.HashMap;
import java.util.Map;
import data.DocumentData;
import kdtree.KDTree;
import methods.*;
import util.EvalUtil;

public class MethodComparison {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		HashMap<Integer, HashMap<Integer, Integer>> docdata = DocumentData.ReadInData("/Users/waltercai/Documents/cse547/hw2/sim_docdata/sim_docdata.mtx", true);
		HashMap<Integer, HashMap<Integer, Integer>> testdata = DocumentData.ReadInData("/Users/waltercai/Documents/cse547/hw2/sim_docdata/test_docdata.mtx", true);
		System.err.println("Number of Documents: " + docdata.keySet().size());
		System.err.println("Number of Test Documents: " + testdata.keySet().size());
		Integer D = 1000;

		int[] ms = {5, 10, 20};
		int[] alphas = {1, 5, 10};
		int depth = 3;

		HashMap<Integer, Double> averageQueryTimeLSH = new HashMap<>();
		HashMap<Integer, Double> averageSimilarityLSH = new HashMap<>();
		for(int m: ms) {
			methods.LocalitySensitiveHash lsh = new LocalitySensitiveHash(docdata, D, m);

			double avdDist = 0.0;
			long startTime = System.currentTimeMillis();
			for(int key : testdata.keySet()) {
				NeighborDistance nn = lsh.NearestNeighbor(testdata.get(key), depth);
				avdDist += nn.distance;
//				java.util.Vector docVec =new java.util.Vector(docdata.get(nn.docId).keySet());
//				java.util.Vector searchVec =new java.util.Vector(testdata.get(key).keySet());
//				Collections.sort(docVec);
//				Collections.sort(searchVec);
//				System.out.println(docVec);
//				System.out.println(searchVec);
//				System.out.println("-----");
			}
			long endTime = System.currentTimeMillis();
			averageQueryTimeLSH.put(m, (endTime - startTime + 0.0) / testdata.keySet().size());
//			averageSimilarityLSH.put(m, - avdDist / testdata.keySet().size());
			averageSimilarityLSH.put(m, avdDist / testdata.keySet().size());
		}
		System.out.println("LSH:");
		System.out.println(averageQueryTimeLSH);
		System.out.println(averageSimilarityLSH);

		KDTree kdt = new KDTree(D);
		double[] key = new double[D];
		for (Map.Entry<Integer, HashMap<Integer, Integer>> document : docdata.entrySet()) {
			for (int i = 0; i < D; i++) {
				key[i] = document.getValue().getOrDefault(i, 0);
			}
			kdt.insert(key, document.getKey());
		}
		HashMap<Integer, Double> averageQueryTimeKDT = new HashMap<>();
		HashMap<Integer, Double> averageSimilarityKDT = new HashMap<>();
		for (int alpha : alphas) {
			double avdDist = 0.0;
			long startTime = System.currentTimeMillis();
			for (int docId : testdata.keySet()) {
				HashMap<Integer, Integer> testDoc = testdata.get(docId);
				for (int i = 0; i < D; i++) {
					key[i] = testDoc.getOrDefault(i, 0);
				}
				Integer nnID = (Integer)kdt.nearest(key, alpha + 0.0);
				NeighborDistance nearestNeighbor = new NeighborDistance(nnID, EvalUtil.Distance(testDoc, docdata.get(nnID)));
				avdDist += nearestNeighbor.distance;
			}
			long endTime = System.currentTimeMillis();
			avdDist = avdDist/testdata.keySet().size();
			averageQueryTimeKDT.put(alpha, (endTime - startTime + 0.0) / testdata.keySet().size());
			averageSimilarityKDT.put(alpha, avdDist);
		}

		System.out.println("KDTree:");
		System.out.println(averageQueryTimeKDT);
		System.out.println(averageSimilarityKDT);

		HashMap<Integer, Double> averageQueryTimeGRP = new HashMap<>();
		HashMap<Integer, Double> averageSimilarityGRP = new HashMap<>();
		for(int m: ms) {
			methods.GaussianRandomProjection grp = new GaussianRandomProjection(docdata, D, m);

			double avdDist = 0.0;
			long startTime = System.currentTimeMillis();
			for(int k : testdata.keySet()) {
				NeighborDistance nn = grp.NearestNeighbor(testdata.get(k), 1.0);
				avdDist += nn.distance;
			}
			long endTime = System.currentTimeMillis();
			averageQueryTimeGRP.put(m, (endTime - startTime + 0.0) / testdata.keySet().size());
			averageSimilarityGRP.put(m, avdDist / testdata.keySet().size());
		}
		System.out.println("GRP:");
		System.out.println(averageQueryTimeGRP);
		System.out.println(averageSimilarityGRP);
	}

}
