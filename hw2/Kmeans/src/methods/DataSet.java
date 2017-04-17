package methods;
import java.util.concurrent.ThreadLocalRandom;

import java.util.*;

public class DataSet {
    private HashMap<Integer, DataPoint> points;
    private int[] currGuesses;
    private double[] currDistances;
    private int dim;
    private int K;
    private int n;
    public double[][] centroids;

    public DataSet(ArrayList<DataPoint> dataList, int _K){
        this.points = new HashMap<>();
        for(int i=0; i<dataList.size(); i++){
            this.points.put(i, dataList.get(i));
        }
        this.n = this.points.size();
        this.K = _K;
        this.currGuesses = new int[this.n];
        this.currDistances = new double[this.n];
        this.centroids = new double[this.K][this.dim];
        this.dim = this.points.get(0).coord.length;
    }

    public void lloyd(){
        this.randomInit();

        int iterCount = 0;

        boolean stable = false;
        while(!stable){
            stable = this.reassign();
            this.reCenter();
            iterCount++;
        }
        this.reCenter();
//        System.out.println("k: " + this.K + ", iterCount: " + iterCount + ", SS: " + this.getAggSS());
    }

    public void kMeansPlusPlus(){
        this.randomInitPP();

        int iterCount = 0;

        boolean stable = false;
        while(!stable){
            stable = this.reassign();
            this.reCenter();
            iterCount++;
        }
        this.reCenter();
//        System.out.println("k: " + this.K + ", iterCount: " + iterCount + ", SS: " + this.getAggSS());
    }

    private boolean reassign(){
        boolean stable = true;

        for(int i=0; i<this.n; i++){
            for(int k=0; k<this.K; k++) {
                double dist = this.points.get(i).getDist(this.centroids[k]);
                if(dist < currDistances[i]){
                    if(this.currGuesses[i] != k){stable = false;}
                    this.currGuesses[i] = k;
                    this.currDistances[i] = dist;
                }
            }
        }

        return stable;
    }

    private void assign(){
        for(int i=0; i<this.n; i++){
            this.currGuesses[i] = 0;
            this.currDistances[i] = this.points.get(i).getDist(this.centroids[0]);

            for(int k=1; k<this.K; k++) {
                double dist = this.points.get(i).getDist(this.centroids[k]);
                if(dist < currDistances[i]){
                    this.currGuesses[i] = k;
                }
            }
        }
    }

    private void reCenter(){
        double[][] coordSum = new double[this.K][this.dim];
        int[] centrCount = new int[this.K];

        for(int i=0; i<this.n; i++){
            int guess = this.currGuesses[i];
            for(int d=0; d<this.dim; d++){
                coordSum[guess][d] += this.points.get(i).coord[d];
            }
            centrCount[guess]++;
        }

        for(int k=0; k<this.K; k++){
            for(int d=0; d<this.dim; d++){
                this.centroids[k][d] = coordSum[k][d] / centrCount[k];
            }
        }
    }

    public void randomInit(){
        int[] range = new int[this.n];
        for(int i=0; i<this.n; i++){
            range[i] = i;
        }
        this.shuffleArray(range);
        for(int k=0; k<this.K; k++){
            this.centroids[k] = this.points.get(range[k]).coord;
        }
        this.assign();
    }

    public void randomInitPP(){
        int[] range = new int[this.n];
        for(int i=0; i<this.n; i++){
            range[i] = i;
        }
        this.shuffleArray(range);
        this.centroids[0] = this.points.get(range[0]).coord;

        for(int k=1; k<this.K; k++){
            double[] minDistArray = new double[this.n];
            for(int i=0; i<this.n; i++) {
                double minDist = this.points.get(i).getDist(this.centroids[0]);
                for (int j = 0; j < k; j++) {
                    double newDist = this.points.get(i).getDist(this.centroids[j]);
                    if (newDist < minDist) {
                        minDist = newDist;
                    }
                }
                minDistArray[i] = minDist;
            }

            RandomCollection<Integer> items = new RandomCollection<Integer>();
            for(int i=0; i<this.n; i++) {
                items.add(minDistArray[i], i);
            }
            int id = items.next();
            this.centroids[k] = this.points.get(id).coord;

        }

        this.assign();
    }

    public double getAggSS(){
        double sumSS = 0.0;
        for(int i=0; i < this.n; i++){
            int guess = this.currGuesses[i];
            sumSS += Math.pow(this.points.get(i).getDist(this.centroids[guess]), 2);
        }

        return sumSS;
    }

    public void printArray(double[][] d){
        int rows = d.length;
        for(int r=0; r<rows; r++) {
            System.out.println(Arrays.toString(d[r]));
        }
    }

    private void shuffleArray(int[] ar)
    {
        // If running on Java 6 or older, use `new Random()` on RHS here
        Random rnd = ThreadLocalRandom.current();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            int a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }

}

