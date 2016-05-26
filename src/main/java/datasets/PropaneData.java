package datasets;


import util.*;

import java.io.*;
import java.util.*;


public class PropaneData {

    private static final String PROPANE_DATA_FILE_PATH = "datasets/propaneData.ser";

    private Map<Float,List<Map<Integer,Integer>>> data;
    private float[] weights;

    public PropaneData() {
        data = deserializePropaneData();

        weights = ProjectUtils.toPrimitiveFloatArray(data.keySet());
    }

    private Map<Float,List<Map<Integer,Integer>>> deserializePropaneData() {
        Map<Float,List<Map<Integer,Integer>>> data = null;
        try {

            ClassLoader classLoader = getClass().getClassLoader();
            File file = new File(classLoader.getResource(PROPANE_DATA_FILE_PATH).getFile());
            FileInputStream fileInputStream = new FileInputStream(file);
            ObjectInputStream in = new ObjectInputStream(fileInputStream);
            data = (Map<Float,List<Map<Integer,Integer>>>) in.readObject();
            in.close();
            fileInputStream.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        return data;
    }

    public List<Map<Integer,Integer>> getFftsForTankWeight(float weight) {
        return data.get(weight);
    }

    public float[] getWeights() {
        return weights;
    }
}
