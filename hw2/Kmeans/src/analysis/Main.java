package analysis;

import methods.DataPoint;
import methods.DataSet;

import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args){
        String dataPath = "/Users/waltercai/Documents/cse547/hw2/2DGaussianMixture.csv";
        ArrayList<DataPoint> dataList = parseCSV(dataPath);

        int[] ks = {2, 3, 5, 10, 15, 20};
        for(int k: ks) {
            DataSet data = new DataSet(dataList, k);
            data.lloyd();
            for(int c=0; c<k; c++) {
                System.out.println(Arrays.toString(data.centroids[c]) + ",");
            }
            System.out.println();
        }

        double[] SSs = new double[20];
        for(int i=0; i<20; i++) {
            DataSet data = new DataSet(dataList, 3);
            data.lloyd();
            for(int c=0; c<3; c++) {
                System.out.println(Arrays.toString(data.centroids[c]) + ",");
            }
            SSs[i] = data.getAggSS();
        }
        System.out.println(Arrays.toString(SSs));

        double[] SSspp = new double[20];
        for(int i=0; i<20; i++) {
            DataSet data = new DataSet(dataList, 3);
            data.kMeansPlusPlus();
            for(int c=0; c<3; c++) {
                System.out.println(Arrays.toString(data.centroids[c]) + ",");
            }
            SSspp[i] = data.getAggSS();
        }
        System.out.println(Arrays.toString(SSspp));
    }

    private static ArrayList<DataPoint> parseCSV(String csvPath){
        BufferedReader br = null;
        String line;
        String cvsSplitBy = ",";
        ArrayList<DataPoint> data = new ArrayList<>();
        boolean firstline = true;

        try {
            br = new BufferedReader(new FileReader(csvPath));
            while ((line = br.readLine()) != null) {
                if(firstline){firstline = false;}
                else {
                    String[] parsedLine = line.split(cvsSplitBy);
                    int label = Integer.parseInt(parsedLine[0]);
                    double[] coord = new double[parsedLine.length - 1];
                    for (int i = 1; i < parsedLine.length; i++) {
                        coord[i - 1] = Double.parseDouble(parsedLine[i]);
                    }

                    data.add(new DataPoint(label, coord));
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
        return data;
    }
}
