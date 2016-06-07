package tests;

import algorithms.*;
import datasets.*;
import util.*;

import java.util.*;

public class BoostingTests {
    public static void testBoostedDecisionStumps_IrisData() {
        BoostingAlgorithm boostingAlgorithm = new BoostingAlgorithm();

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.DecisionStump");

        boostingAlgorithm.setParams(params);

        System.out.println("\niris datset (boosted decision stumps):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(irisDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));
    }

    public static void testBoostedC45_IrisData() {
        BoostingAlgorithm boostingAlgorithm = new BoostingAlgorithm();

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.J48");

        boostingAlgorithm.setParams(params);

        System.out.println("\niris datset (boosted C4.5 tree):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(irisDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));
    }

    public static void testBoostedDecisionStumps_PropaneData() {
        BoostingAlgorithm boostingAlgorithm = new BoostingAlgorithm();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.DecisionStump");

        boostingAlgorithm.setParams(params);

        System.out.println("\npropane dataSet (boosted C4.5):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));

    }

    public static void testBoostedC45_PropaneData() {
        BoostingAlgorithm boostingAlgorithm = new BoostingAlgorithm();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(BoostingAlgorithm.KEY_ALGORITHM_CLASS_NAME, "weka.classifiers.trees.DecisionStump");

        boostingAlgorithm.setParams(params);

        System.out.println("\npropane dataSet (boosted C4.5):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, boostingAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));

    }

    public static void learningCurveBoostedC45_PropaneData() {
        BoostingAlgorithm boostingAlgorithm;

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

    public static void learningCurveBoostedDecisionStumps_IrisData() {
        BoostingAlgorithm boostingAlgorithm;

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        System.out.println("\niris dataSet learning curve (boosted Decision Stumps):");
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
