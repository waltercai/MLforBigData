package analysis;

import methods.DataSet;
import java.util.Arrays;

public class BBC {
    public static void main(String[] args){
        String mtxPath = "/Users/waltercai/Documents/cse547/hw2/bbc_data/bbc.mtx";
        String centersPath = "/Users/waltercai/Documents/cse547/hw2/bbc_data/bbc.centers";
        String classesPath = "/Users/waltercai/Documents/cse547/hw2/bbc_data/bbc.classes";

        int k = 5;
        DataSet data = new DataSet(mtxPath, classesPath, k);
        data.kmeans(centersPath);

        for(int c=0; c<k; c++) {
            System.out.println(Arrays.toString(data.centroids[c]) + ",");
        }
        System.out.println();
    }
}
