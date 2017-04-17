package methods;

import java.util.NavigableMap;
import java.util.TreeMap;
import java.util.Random;

/**
 * I have borrowed this from Peter Lawrey <http://stackoverflow.com/users/57695/peter-lawrey>
 * so that I can run a weighted random sampling in order to implement k-means++
 */
public class RandomCollection<E> {
    private final NavigableMap<Double, E> map = new TreeMap<Double, E>();
    private final Random random;
    private double total = 0;

    public RandomCollection() {
        this(new Random());
    }

    public RandomCollection(Random random) {
        this.random = random;
    }

    public void add(double weight, E result) {
        if (weight <= 0) return;
        total += weight;
        map.put(total, result);
    }

    public E next() {
        double value = random.nextDouble() * total;
        return map.ceilingEntry(value).getValue();
    }
}