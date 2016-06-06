package algorithms;

import datasets.*;
import org.junit.*;
import util.*;

import java.util.*;

public class TestBoostingAlgorithm {

    private BoostingAlgorithm boostingAlgorithm;

    @Before
    public void setUp() {
        boostingAlgorithm = new BoostingAlgorithm();
    }

    @Test
    public void testBoostingOnPropaneData() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.DecisionStump");

        boostingAlgorithm.setParams(params);

        System.out.println("\npropane dataSet:");
        CrossValidation.crossValidate(propaneDataSet, 30, boostingAlgorithm);
    }

    @Test
    public void testBoostingOnIrisData() {
        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.DecisionStump");

        boostingAlgorithm.setParams(params);

        System.out.println("\niris datset:");
        CrossValidation.crossValidate(irisDataSet, 10, boostingAlgorithm);
    }
}
