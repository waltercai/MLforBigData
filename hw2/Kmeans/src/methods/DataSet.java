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

    public void lloyd(){
        double[][] coord = this.randomInit();

        int[] newGuess = new int[this.n];
        int[] oldGuess = null;
        for(int i=0; i<this.n; i++){newGuess[i] = this.K;}
        boolean stable = false;

        int iterCount = 0;

        stable = false;
        while (!stable){
            oldGuess = newGuess;
            newGuess = this.reassign(coord);
            coord = this.reCenter(newGuess);

            stable = true;
            for(int i=0; i<newGuess.length; i++){if(newGuess[i] != oldGuess[i]){
                stable = false;
            }}
            iterCount++;
        }

        this.finalGuesses = newGuess;
        this.finalDistances = new double[this.n];
        this.centroids = coord;
        for(int i=0; i<this.n; i++){
            this.finalDistances[i] = this.points.get(i).getDist(coord[this.finalGuesses[i]]);
        }
//        System.out.println("k: " + this.K + ", iterCount: " + iterCount + ", SS: " + this.getAggSS());
    }

    public void kMeansPlusPlus() {
        double[][] coord = this.randomInitPP();

        int[] newGuess = new int[this.n];
        int[] oldGuess = null;
        for(int i=0; i<this.n; i++){newGuess[i] = this.K;}
        boolean stable = false;

        int iterCount = 0;

        stable = false;
        while (!stable){
            oldGuess = newGuess;
            newGuess = this.reassign(coord);
            coord = this.reCenter(newGuess);

            stable = true;
            for(int i=0; i<newGuess.length; i++){if(newGuess[i] != oldGuess[i]){
                stable = false;
            }}
            iterCount++;
        }

        this.finalGuesses = newGuess;
        this.finalDistances = new double[this.n];
        this.centroids = coord;
        for(int i=0; i<this.n; i++){
            this.finalDistances[i] = this.points.get(i).getDist(coord[this.finalGuesses[i]]);
        }
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

    public double[][] randomInit(){
        double[][] coord = new double[this.K][this.dim];
        int[] range = new int[this.n];
        for(int i=0; i<this.n; i++){
            range[i] = i;
        }
        this.shuffleArray(range);
        for(int k=0; k<this.K; k++){
            coord[k] = this.points.get(range[k]).coord;
        }

        return(coord);
    }

    public double[][] randomInitPP(){
        double[][] coord = new double[this.K][this.dim];
        int[] range = new int[this.n];
        for(int i=0; i<this.n; i++){
            range[i] = i;
        }
        this.shuffleArray(range);
        coord[0] = this.points.get(range[0]).coord;

        for(int k=1; k<this.K; k++){
            double[] minDistArray = new double[this.n];
            for(int i=0; i<this.n; i++) {
                double minDist = this.points.get(i).getDist(coord[0]);
                for (int j = 1; j < k; j++) {
                    double newDist = this.points.get(i).getDist(coord[j]);
                    if (newDist < minDist) {
                        minDist = newDist;
                    }
                }
                minDistArray[i] = minDist;
            }

            RandomCollection<Integer> items = new RandomCollection<>();
            for(int i=0; i<this.n; i++) {
                items.add(minDistArray[i], i);
            }
            int id = items.next();
            coord[k] = this.points.get(id).coord;
        }
        return(coord);
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

