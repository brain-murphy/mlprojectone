package util;

import java.util.*;

public class ProjectUtils {
    public static float[] toPrimitiveFloatArray(Collection<Float> floatCollection) {
        float[] primitiveArray = new float[floatCollection.size()];
        Iterator iterator = floatCollection.iterator();

        for (int i = 0; i < floatCollection.size(); i++) {
            primitiveArray[i] = (float) iterator.next();
        }

        return primitiveArray;
    }

    public static double mean(double[] numbers) {
        double sum = 0;
        for (double number : numbers) {
            sum += number;
        }
        return sum / numbers.length;
    }
}
