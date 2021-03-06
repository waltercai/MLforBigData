package edu.uw.cs.biglearn.hadoop.kmeans.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import edu.uw.cs.biglearn.hadoop.kmeans.Cluster;
import edu.uw.cs.biglearn.hadoop.kmeans.Document;
import edu.uw.cs.biglearn.util.MathUtil;

/**
 * Update the cluster center and compute the with-in class distances
 * @author haijieg
 */
public class KmeansReducer extends MapReduceBase implements Reducer<IntWritable, Text, Text, Text>{
	
	@Override
	public void reduce(IntWritable key, Iterator<Text> values,
			OutputCollector<Text, Text> out, Reporter reporter) throws IOException {
		
		// new cluster centroid
		Cluster c = new Cluster();
		// squared distance within cluster
		double sqdist = 0.0;
	
		/**
		 * TODO: Your code goes in here.
		 */
	
		// Output the cluster center into file: clusteri
		out.collect(new Text("cluster" + c.id), new Text(c.toString()));
		
		// Output the within distance into file: distancei
		out.collect(new Text("distance" + c.id), new Text(c.id + "|" + String.valueOf(sqdist)));
	}
	

}
