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
        double[] results = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testIrisData() {
        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.DecisionStump");

        boostingAlgorithm.setParams(params);

        System.out.println("\niris datset (boosted decision stumps):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(irisDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testPropaneDataC45() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.J48");

        boostingAlgorithm.setParams(params);

        System.out.println("\npropane dataSet (boosted C4.5 tree):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testIrisDataC45() {
        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.J48");

        boostingAlgorithm.setParams(params);

        System.out.println("\niris datset (boosted C4.5 tree):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(irisDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testPropaneDataRepTree() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.REPTree");

        boostingAlgorithm.setParams(params);

        System.out.println("\npropane dataSet (boosted REP trees):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testIrisDataRepTree() {
        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.REPTree");

        boostingAlgorithm.setParams(params);

        System.out.println("\niris datset (boosted REP trees):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(irisDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void learningCurveC45PropaneData() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        System.out.println("\npropane dataSet learning curve (boosted C4.5 tree):");
        System.out.println("boostingIterations,trainingError,crossValidationError");
        for (int iterations = 2; iterations < 100; iterations++) {
            boostingAlgorithm = new BoostingAlgorithm();

            Map<String, Object> params = new HashMap<>();

            params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.DecisionStump");

            params.put(BoostingAlgorithm.KEY_ITERATIONS, iterations);

            boostingAlgorithm.setParams(params);

            double[] results = ProjectUtils.crossValidate(propaneDataSet, 30, boostingAlgorithm);

            System.out.println(Integer.toString(iterations) + "," + results[4] + "," + String.format("%.3f", results[0]));
        }
    }

    @Test
    public void learningCurveDecisionStumpsIrisData() {
        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        System.out.println("\niris dataSet learning curve (boosted C4.5 tree):");
        System.out.println("boostingIterations,trainingError,crossValidationError");
        for (int iterations = 2; iterations < 100; iterations++) {
            boostingAlgorithm = new BoostingAlgorithm();

            Map<String, Object> params = new HashMap<>();

            params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.DecisionStump");

            params.put(BoostingAlgorithm.KEY_ITERATIONS, iterations);

            boostingAlgorithm.setParams(params);

            double[] results = ProjectUtils.leaveOneOutCrossValidate(irisDataSet, boostingAlgorithm);

            System.out.println(Integer.toString(iterations) + "," + results[4] + "," + String.format("%.3f", results[0]));
        }
    }
}
