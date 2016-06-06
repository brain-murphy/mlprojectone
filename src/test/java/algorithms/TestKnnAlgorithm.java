package algorithms;

import datasets.*;
import org.junit.*;
import util.*;

import java.util.*;

public class TestKnnAlgorithm {
    private KNearestNeighborsAlgorithm knnAlgorithm;

    @Before
    public void setUp() {
        knnAlgorithm = new KNearestNeighborsAlgorithm();
    }

    @Test
    public void testKnnOnPropaneData() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(KNearestNeighborsAlgorithm.KEY_K, 5);

        CrossValidation.crossValidate(propaneDataSet, 30, knnAlgorithm);
    }

    @Test
    public void testKnnOnIrisData() {
        DataSet<IrisInstance> irisInstanceDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(KNearestNeighborsAlgorithm.KEY_K, 5);

        CrossValidation.crossValidate(irisInstanceDataSet, 10, knnAlgorithm);
    }
}
