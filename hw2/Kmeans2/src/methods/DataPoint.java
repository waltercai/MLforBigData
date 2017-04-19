package methods;

import java.util.Arrays;
import java.util.HashMap;

public class DataPoint {
    public int label;
    public HashMap<Integer, Double> coord;

    public DataPoint(int _label, HashMap<Integer, Double> _coord){
        this.label = _label;
        this.coord = _coord;
    }

    public double getDist(double[] coord2){
        double dist = 0.0;
        for(int t=0; t<coord2.length; t++){
            dist += Math.pow(coord2[t] - this.coord.getOrDefault(t,0.0), 2);
        }
        return dist;
    }
}
