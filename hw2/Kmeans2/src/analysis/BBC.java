package analysis;

import methods.DataPoint;
import methods.DataSet;

import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class BBC {
    public static void main(String[] args){
        String mtxPath = "/Users/waltercai/Documents/cse547/hw2/bbc/bbc.mtx";
        String centersPath = "/Users/waltercai/Documents/cse547/hw2/bbc/bbc.centers";
        String classesPath = "/Users/waltercai/Documents/cse547/hw2/bbc/bbc.classes";
        DataSet data;

        int k = 5;
        data = new DataSet(mtxPath, classesPath);
        data.kmeans(centersPath);
        System.out.println();
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
