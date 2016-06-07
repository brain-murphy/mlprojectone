package tests;

import algorithms.*;
import datasets.*;
import util.*;

import java.util.*;

public class DecisionTreeTests {
    public static void testC45_PropaneData() {

        DecisionTreeAlgorithm decisionTreeAlgorithm = new DecisionTreeAlgorithm();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, false);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\npropane dataset (C4.5 tree):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));
    }

    public static void testC45REP_PropaneData() {

        DecisionTreeAlgorithm decisionTreeAlgorithm = new DecisionTreeAlgorithm();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, true);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\npropane dataset (C4.5 with reduced error pruning):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));
    }

    public static void testC45AgressivePruning_PropaneData() {

        DecisionTreeAlgorithm decisionTreeAlgorithm = new DecisionTreeAlgorithm();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        double pruningConfidence = 0.1;

        params.put(DecisionTreeAlgorithm.KEY_PRUNING_CONFIDENCE, pruningConfidence);

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, false);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\npropane dataset (C4.5 with aggressive pruning):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));
    }

    public static void testC45BinarySplits_PropaneData() {
        DecisionTreeAlgorithm decisionTreeAlgorithm = new DecisionTreeAlgorithm();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, false);

        params.put(DecisionTreeAlgorithm.KEY_ONLY_BINARY_SPLITS, true);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\npropane dataset (C4.5 with only binary splits):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));
    }



    public static void testC45_IrisData() {

        DecisionTreeAlgorithm decisionTreeAlgorithm = new DecisionTreeAlgorithm();

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, false);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\niris dataset (C4.5 tree):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(irisDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));
    }

    public static void testC45REP_IrisData() {

        DecisionTreeAlgorithm decisionTreeAlgorithm = new DecisionTreeAlgorithm();

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, true);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\niris dataset (C4.5 with reduced error pruning):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(irisDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));
    }

    public static void testC45AgressivePruning_IrisData() {

        DecisionTreeAlgorithm decisionTreeAlgorithm = new DecisionTreeAlgorithm();

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        double pruningConfidence = 0.1;

        params.put(DecisionTreeAlgorithm.KEY_PRUNING_CONFIDENCE, pruningConfidence);

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, false);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\niris dataset (C4.5 with aggressive pruning):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(irisDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));
    }

    public static void testC45BinarySplits_IrisData() {
        DecisionTreeAlgorithm decisionTreeAlgorithm = new DecisionTreeAlgorithm();

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, false);

        params.put(DecisionTreeAlgorithm.KEY_ONLY_BINARY_SPLITS, true);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\niris dataset (C4.5 with only binary splits):");
        double[] results = ProjectUtils.leaveOneOutCrossValidate(irisDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]));
    }

    public static void learningCurveC45_PropaneData() {
        DecisionTreeAlgorithm decisionTreeAlgorithm = new DecisionTreeAlgorithm();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, false);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\npropane datset learning curve (C4.5 tree):");
        ProjectUtils.printLearningCurve(propaneDataSet, decisionTreeAlgorithm);
    }

    public static void learningCurveC45_IrisData() {
        DecisionTreeAlgorithm decisionTreeAlgorithm = new DecisionTreeAlgorithm();

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, true);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\niris datset learning curve (C4.5 tree):");
        ProjectUtils.printLearningCurve(irisDataSet, decisionTreeAlgorithm);
    }
}

