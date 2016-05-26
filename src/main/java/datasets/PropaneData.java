package datasets;


import util.*;

import java.io.*;
import java.util.*;


public class PropaneData {

    private static final String PROPANE_DATA_FILE_PATH = "datasets/propaneData.ser";

    private Map<Float,List<Map<Integer,Integer>>> data;
    private float[] weights;

    public static PropaneData readPropaneDataFromResources() {
        PropaneData propaneData = new PropaneData(deserializePropaneData());

        return propaneData;
    }

    private static Map<Float,List<Map<Integer,Integer>>> deserializePropaneData() {
        Map<Float,List<Map<Integer,Integer>>> data = null;
        try {
            FileInputStream fileIn = new FileInputStream(PROPANE_DATA_FILE_PATH);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            data = (Map<Float,List<Map<Integer,Integer>>>) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        return data;
    }

    private PropaneData(Map<Float,List<Map<Integer,Integer>>> pData) {
        data = pData;

        weights = ProjectUtils.toPrimitiveFloatArray(data.keySet());
    }

    public List<Map<Integer,Integer>> getFftsForTankWeight(float weight) {
        return data.get(weight);
    }

    public float[] getWeights() {
        return weights;
    }
}
