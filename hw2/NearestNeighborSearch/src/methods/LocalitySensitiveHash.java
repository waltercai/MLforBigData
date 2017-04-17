package methods;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import util.EvalUtil;

public class LocalitySensitiveHash {

	// Projection vectors
	public double[][] projVectors;
	// Hash of projected bin to document ids
	public HashMap<Integer, HashSet<Integer>> hashedDocuments;
	// reference to the original documents
	private HashMap<Integer, HashMap<Integer, Integer>> docs;
	// Dimensionality of the data
	private Integer D;
	// Number of projections to use (dimensionality of LSH)
	private Integer m;

	public LocalitySensitiveHash(
			HashMap<Integer, HashMap<Integer, Integer>> documents,
			Integer D,
			Integer m) throws Exception {
		
		this.docs = documents;
		this.D = D;
		this.m = m;
		projVectors = Helper.CreateProjectionVectors(D, m);
		BuildHashedDocuments();
	}
	
	/**Gets the (approximate) nearest neighbor to the given document
	 * @param document - the document to find the nearest neighbor for
	 * @param depth - the maximum number of bits to change concurrently
	 * @return
	 */
	public NeighborDistance NearestNeighbor(HashMap<Integer, Integer> document, Integer depth) throws Exception {
		Boolean[] hashedDoc = HashDocument(document);
		NeighborDistance nearest = NearestNeighbor(document, hashedDoc, null, depth, 0);
		return nearest;
	}
		
	/**
	 * @param document - the document to find the nearest neighbor for
	 * @param hashedDoc
	 * @param curNearest
	 * @param depth
	 * @param nextIndex
	 */
	private NeighborDistance NearestNeighbor(
			HashMap<Integer, Integer> document,
			Boolean[] hashedDoc,
			NeighborDistance curNearest,
			Integer depth,
			Integer nextIndex) throws Exception {
		
		if ( depth < 0 ) {
			return curNearest;
		}
		if ( null == curNearest ) {
			curNearest = new NeighborDistance(0, Double.MAX_VALUE);
		}
		CheckBin(document, hashedDoc, curNearest);
		if ( depth > 0 ) {
			// check the bins one away from the current bin
			// if we still have more depth to go
			for (int j = nextIndex; j < m; j++) {
				hashedDoc[j] = !hashedDoc[j];
				NearestNeighbor(document, hashedDoc, curNearest, depth-1, j+1);
				hashedDoc[j] = !hashedDoc[j];				
			}
		}
		return curNearest;
	}
	
	/**
	 * Checks the documents that are hashed to the given bin
	 * and updates with the nearest neighbor found
	 * @param document - the document to find the nearest neighbor for
	 * @param bin - the bin to search
	 * @param curNearest - the current nearest neighbor
	 */
	private void CheckBin(
			HashMap<Integer, Integer> document, 
			Boolean[] bin,
			NeighborDistance curNearest) throws Exception {
		int binInt = this.ConvertBooleanArrayToInteger(bin);
		if(this.hashedDocuments.containsKey(binInt)){
			for (int curr : this.hashedDocuments.get(binInt)) {
				HashMap<Integer, Integer> currDoc = this.docs.get(curr);
				double currDist = EvalUtil.Distance(currDoc, document);
				if (currDist < curNearest.distance) {
					curNearest.docId = curr;
					curNearest.distance = currDist;
				}
			}
		}
	}

	/**
	 * generates the true cos(theta) for two vectors
	 * @param search - first document (same vector from earlier version so no need to normalize again)
	 * @param cand - second document (should normalize)
	 * @return cosTheta - cosine of the angle between the two input vectors
	 */
	public double halfNormedCosineTheta(HashMap<Integer, Integer> search, HashMap<Integer, Integer> cand){
		int dotProd = 0;
		for (int key : search.keySet()) {
			dotProd += search.get(key) * cand.getOrDefault(key, 0);
		}
		double norm2 = 0.0;
		for(double val: cand.values()){ norm2 += Math.pow(val, 2);}
		norm2 = Math.sqrt(norm2);
		return - dotProd / norm2;
	}
	
	/**
	 * Builds the hashtable of documents
	 */
	private void BuildHashedDocuments() throws Exception {
		hashedDocuments = new HashMap<Integer, HashSet<Integer>>();
		Integer bin;
		for (Entry<Integer, HashMap<Integer, Integer>> entry : docs.entrySet()) {
			bin = GetBin(entry.getValue());
			if ( !hashedDocuments.containsKey(bin) ) {
				hashedDocuments.put(bin, new HashSet<Integer>());
			}
			hashedDocuments.get(bin).add(entry.getKey());
		}		
	}
	
	/**
	 * Gets the bin where a certain document should be stored
	 * @param document
	 * @return
	 */
	private Integer GetBin(HashMap<Integer, Integer> document) throws Exception {
		return ConvertBooleanArrayToInteger(HashDocument(document));
	}
	
	/**Hashes a document to a boolean array using the set of projection vectors
	 * @param document
	 * @return
	 */
	public Boolean[] HashDocument(HashMap<Integer, Integer> document) throws Exception {
		Boolean[] hashedDoc = new Boolean[m];
		for(int i=0; i < this.m; i++){
			hashedDoc[i] = this.ProjectDocument(document, this.projVectors[i]);
		}
		return hashedDoc;
	}
	
	/**Projects a document onto a projection vector for a boolean result
	 * @param document
	 * @param projVector
	 * @return false if the projection is negative, true if the projection is positive
	 */
	private Boolean ProjectDocument(HashMap<Integer, Integer> document, double[] projVector) throws Exception {
		double dotProd = 0.0;
		for(int i: document.keySet()){
			dotProd += document.get(i) * projVector[i-1];
		}
		return dotProd < 0;
	}
	
	/**Converts a boolean array to the corresponding integer value
	 * @param boolArray
	 * @return
	 */
	private Integer ConvertBooleanArrayToInteger(Boolean[] boolArray) {
		Integer value = 0;
		for (int i = 0; i < boolArray.length; i++) {
			value += boolArray[i] ? (int)Math.round(Math.pow(2, i)) : 0;
		}
		return value;
	}

}
