package methods;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class DataSet {
    public int[] finalGuesses;
    private double[] finalDistances;
    public double[][] centroids;
    private ArrayList<DataPoint> points;
    private int dim;
    private int K;
    private int n;

    public DataSet(String mtxPath, String classesPath, int _K){
        BufferedReader br = null;
        String line;
        String cvsSplitBy = " ";

        this.points = null;
        this.K = _K;

        try {
            br = new BufferedReader(new FileReader(mtxPath));
            int lineCounter = 0;
            int[] preIDF = null;
            while ((line = br.readLine()) != null) {
                if(lineCounter == 0){}
                else if(lineCounter == 1){
                    String[] parsedLine = line.split(cvsSplitBy);
                    this.dim = Integer.parseInt(parsedLine[0]);
                    this.n = Integer.parseInt(parsedLine[1]);
                    this.points = new ArrayList<DataPoint>();
                    for(int i=0; i<this.n; i++){
                        this.points.add(new DataPoint());
                    }
                    preIDF = new int[this.dim];
                }
                else {
                    String[] parsedLine = line.split(cvsSplitBy);
                    int term = Integer.parseInt(parsedLine[0]) - 1;
                    int docid = Integer.parseInt(parsedLine[1]) - 1;
                    double freq = Double.parseDouble(parsedLine[2]);

                    preIDF[term]++;

                    this.points.get(docid).coord.put(term, freq);
                }
                lineCounter++;
            }
            for(int i=0; i<this.n; i++){
                Double docMaxFreq = Collections.max(this.points.get(i).coord.values());
                for(int t: this.points.get(i).coord.keySet()){
                    Double oldVal = this.points.get(i).coord.get(t);
                    Double newVal = oldVal/docMaxFreq * Math.log((0.0 + this.points.size())/ preIDF[t]);
                    this.points.get(i).coord.put(t, newVal);
                }
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        try {
            br = new BufferedReader(new FileReader(classesPath));
            int lineCounter = 0;
            while ((line = br.readLine()) != null) {
                String[] parsedLine = line.split(cvsSplitBy);
                int label = Integer.parseInt(parsedLine[1]);

                this.points.get(lineCounter).label = label;
                lineCounter++;
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }


    }

    public void kmeans(String centersPath){
        double[][] coord = this.getCenters(centersPath);

        int[] newGuess = new int[this.n];
        for(int i=0; i<this.n; i++){newGuess[i] = this.K;}

        for(int j=0; j<5; j++){
            newGuess = this.reassign(coord);
            coord = this.reCenter(newGuess);

            System.out.println(this.getLoss(newGuess));
        }

        this.finalGuesses = newGuess;
        this.finalDistances = new double[this.n];
        this.centroids = coord;
        for(int i=0; i<this.n; i++){
            this.finalDistances[i] = this.points.get(i).getDist(coord[this.finalGuesses[i]]);
        }

//        System.out.println("k: " + this.K + ", iterCount: " + iterCount + ", SS: " + this.getAggSS());
    }

    private int getLoss(int[] guess){
        int loss = 0;
        for(int i=0; i<this.n; i++){
            if(guess[i] != this.points.get(i).label){loss++;}
        }
        return loss;
    }

    private double[][] getCenters(String centersPath){
        double[][] centers = new double[this.K][this.dim];

        BufferedReader br = null;
        String line;
        String cvsSplitBy = " ";

        try {
            br = new BufferedReader(new FileReader(centersPath));
            int lineCounter = 0;
            while ((line = br.readLine()) != null) {
                String[] parsedLine = line.split(cvsSplitBy);
                for(int i=0; i<parsedLine.length; i++){
                    centers[lineCounter][i] = Double.parseDouble(parsedLine[i]);
                }
                lineCounter++;
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return centers;
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
        int[] centerCount = new int[this.K];

        for(int i=0; i<this.n; i++){
            int guess = guesses[i];
            for(int d=0; d<this.dim; d++){
                coordSum[guess][d] += this.points.get(i).coord.getOrDefault(d,0.0);
            }
            centerCount[guess]++;
        }

        for(int k=0; k<this.K; k++){
            for(int d=0; d<this.dim; d++){
                coordSum[k][d] = coordSum[k][d] / centerCount[k];
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

    public void preview(){
        for(int i=0; i<5; i++){
            System.out.println(points.get(i).label);
            System.out.println(points.get(i).coord);
        }
    }
}

