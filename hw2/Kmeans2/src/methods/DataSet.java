package methods;
import java.util.concurrent.ThreadLocalRandom;

import java.util.*;

public class DataSet {
    public int[] finalGuesses;
    private double[] finalDistances;
    public double[][] centroids;
    private HashMap<Integer, DataPoint> points;
    private int dim;
    private int K;
    private int n;

    public DataSet(ArrayList<DataPoint> dataList, int _K){
        this.points = new HashMap<>();
        for(int i=0; i<dataList.size(); i++){
            this.points.put(i, dataList.get(i));
        }
        this.n = this.points.size();
        this.K = _K;
        this.dim = this.points.get(0).coord.length;
    }

    public void kmeans(){

    }

    private int[] reassign(double[][] coord){
        int[] newGuesses = new int[this.n];
        double[] currDistances = new double[this.n];

        for (int i=0; i<this.n; i++) {
            currDistances[i] = Double.MAX_VALUE;
        }

        for(int i=0; i<this.n; i++){
            for(int k=0; k<this.K; k++) {
                double dist = this.points.get(i).getDist(coord[k]);
                if(dist < currDistances[i]){
                    newGuesses[i] = k;
                    currDistances[i] = dist;
                }
            }
        }

        return newGuesses;
    }

    private double[][] reCenter(int[] guesses){
        double[][] coordSum = new double[this.K][this.dim];
        int[] centrCount = new int[this.K];

        for(int i=0; i<this.n; i++){
            int guess = guesses[i];
            for(int d=0; d<this.dim; d++){
                coordSum[guess][d] += this.points.get(i).coord[d];
            }
            centrCount[guess]++;
        }

        for(int k=0; k<this.K; k++){
            for(int d=0; d<this.dim; d++){
                coordSum[k][d] = coordSum[k][d] / centrCount[k];
            }
        }

        return(coordSum);
    }

    public double getAggSS(){
        double sumSS = 0.0;
        for(int i=0; i < this.n; i++){
            sumSS += this.finalDistances[i];
        }

        return sumSS;
    }

    public void printArray(double[][] d){
        int rows = d.length;
        for(int r=0; r<rows; r++) {
            System.out.println(Arrays.toString(d[r]));
        }
    }
}

