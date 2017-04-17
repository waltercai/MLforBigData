package util;

import java.util.HashMap;
import java.util.Map.Entry;

public class EvalUtil {

	public static double Distance( Double[] x1, Double[] x2 ) {
		double dist = 0.0;
		for (int i = 0; i < x1.length; i++) {
			dist += Math.pow(x1[i]-x2[i], 2);
		}
		return Math.sqrt(dist);
	}

	/**
	 * Calculates the distance between two documents
	 * @param doc1
	 * @param doc2
	 * @return
	 */
	public static Double Distance(HashMap<Integer, Integer> doc1, HashMap<Integer, Integer> doc2) {
		Double dist = 0.0;
		for (Entry<Integer, Integer> entry : doc1.entrySet()) {
			dist += doc2.containsKey(entry.getKey()) ?
					Math.pow(entry.getValue() - doc2.get(entry.getKey()), 2) :
						Math.pow(entry.getValue(), 2);
		}
		for (Entry<Integer, Integer> entry : doc2.entrySet()) {
			dist += doc1.containsKey(entry.getKey()) ?
					0.0 : Math.pow(entry.getValue(), 2);
		}
		return Math.sqrt(dist);
	}
	
	public static double Distance( double[] x1, HashMap<Integer, Integer> doc ) {
		double dist = 0.0;
		for (int i = 0; i < x1.length; i++) {
			if ( doc.containsKey(i+1) ) {
				dist += Math.pow(x1[i] - doc.get(i+1), 2);
			}
			else {
				dist += Math.pow(x1[i], 2);
			}
		}
		return Math.sqrt(dist);
	}

}
