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
    public void testPropaneData() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.DecisionStump");

        boostingAlgorithm.setParams(params);

        System.out.println("\npropane dataSet (boosted decision stumps):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(propaneDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testIrisData() {
        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.DecisionStump");

        boostingAlgorithm.setParams(params);

        System.out.println("\niris datset (boosted decision stumps):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(irisDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testPropaneDataC45() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.J48");

        boostingAlgorithm.setParams(params);

        System.out.println("\npropane dataSet (boosted C4.5 tree):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(propaneDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testIrisDataC45() {
        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.J48");

        boostingAlgorithm.setParams(params);

        System.out.println("\niris datset (boosted C4.5 tree):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(irisDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testPropaneDataRepTree() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.REPTree");

        boostingAlgorithm.setParams(params);

        System.out.println("\npropane dataSet (boosted REP trees):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(propaneDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testIrisDataRepTree() {
        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.REPTree");

        boostingAlgorithm.setParams(params);

        System.out.println("\niris datset (boosted REP trees):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(irisDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }
}
