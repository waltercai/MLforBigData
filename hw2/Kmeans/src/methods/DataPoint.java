package methods;

import java.util.Arrays;

public class DataPoint {
    public int label;
    public double[] coord;

    public DataPoint(int _label, double[] _coord){
        this.label = _label;
        this.coord = _coord;
    }

    public double getDist(double[] coord2){
        if(coord2.length != this.coord.length){
            System.exit(2);
        }
        double dist = 0.0;
        for(int i=0; i<this.coord.length; i++){
            dist += Math.pow(this.coord[i] - coord2[i], 2);
        }
        return dist;
    }

    public void print(){
        System.out.println("label: " + this.label + ", coord: " + Arrays.toString(this.coord));
    }
}
