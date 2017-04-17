package methods;

import java.util.HashMap;

import util.EvalUtil;

import kdtree.KDTree;

public class GaussianRandomProjection {
	// Projection vectors
	public double[][] projVectors;
	// Hash of projected bin to document ids
	public KDTree kdt;
	// reference to the original documents
	private HashMap<Integer, HashMap<Integer, Integer>> docs;
	// Dimensionality of the data
	private Integer D;
	// Number of projections to use (dimensionality of LSH)
	private Integer m;

	public GaussianRandomProjection(
			HashMap<Integer, HashMap<Integer, Integer>> documents,
			Integer D,
			Integer m) throws Exception {
		
		this.docs = documents;
		this.D = D;
		this.m = m;
		projVectors = Helper.CreateProjectionVectors(D, m);
		kdt = new KDTree(m);
		for (Integer docId : documents.keySet()) {
			kdt.insert(HashDocument(documents.get(docId)), docId);
		}		
	}

	/**Gets the (approximate) nearest neighbor to the given document
	 * @param document
	 * @param alpha for the KD Tree
	 * @return
	 */
	public NeighborDistance NearestNeighbor(HashMap<Integer, Integer> document, double alpha) throws Exception {
		double[] hashedDoc = HashDocument(document);
		Integer nearestId = (Integer)kdt.nearest(hashedDoc, alpha);
		NeighborDistance nearest = new NeighborDistance(nearestId, EvalUtil.Distance(document, docs.get(nearestId)));
		return nearest;
	}
	
	/**Hashes a document to a double array using the set of projection vectors
	 * @param document
	 * @return
	 */
	public double[] HashDocument(HashMap<Integer, Integer> document) throws Exception {
		double[] hashedDoc = new double[m];
		for(int i = 0; i < this.m; i++){
			hashedDoc[i] = this.ProjectDocument(document, this.projVectors[i]);
		}
		return hashedDoc;
	}
	
	/**Projects a document onto a projection vector, returning the double value
	 * @param document
	 * @param projVector
	 * @return
	 */
	private static double ProjectDocument(HashMap<Integer, Integer> document, double[] projVector) throws Exception {
		double dotProd = 0.0;
		for(int key: document.keySet()){
			dotProd += document.get(key) * projVector[key-1];
		}
		return dotProd;
	}

}
