package tests;

import algorithms.*;
import datasets.*;
import util.*;

import java.util.*;

public class KNNTests {
    public static void crossValidateForK_IrisData() {
        KNearestNeighborsAlgorithm knnAlgorithm;

        DataSet<IrisInstance> irisInstanceDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        int highestKToTest = 20;

        double[] errors = new double[highestKToTest];

        System.out.println("\nLOOCV for best k on iris data:");
        for (int k = 1; k <= highestKToTest; k++) {

            knnAlgorithm = new KNearestNeighborsAlgorithm();

            params.put(KNearestNeighborsAlgorithm.KEY_K, k);

            knnAlgorithm.setParams(params);

            errors[k - 1] = ProjectUtils.leaveOneOutCrossValidate(irisInstanceDataSet, knnAlgorithm)[0];

            System.out.println("k=" + k + " error:" + errors[k - 1]);
        }
    }

    public static void learningCurve7NN_IrisData() {

        KNearestNeighborsAlgorithm knnAlgorithm = new KNearestNeighborsAlgorithm();

        DataSet<IrisInstance> irisInstanceDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        int k = 7;

        params.put(KNearestNeighborsAlgorithm.KEY_K, k);

        knnAlgorithm.setParams(params);

        System.out.println("iris dataSet learning curve (KNN with k=7):");
        ProjectUtils.printLearningCurve(irisInstanceDataSet, knnAlgorithm);
    }

    public static void crossValidateForK_PropaneData() {
        KNearestNeighborsAlgorithm knnAlgorithm;
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        int highestKToTest = 20;

        double[] errors = new double[highestKToTest];

        System.out.println("\nLOOCV for best k on propane data:");
        for (int k = 1; k <= highestKToTest; k++) {

            knnAlgorithm = new KNearestNeighborsAlgorithm();

            params.put(KNearestNeighborsAlgorithm.KEY_K, k);

            knnAlgorithm.setParams(params);

            errors[k - 1] = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, knnAlgorithm)[0];

            System.out.println("k=" + k + " error:" + errors[k - 1]);
        }
    }

    public static void learningCurve1NN_PropaneData() {
        KNearestNeighborsAlgorithm knnAlgorithm = new KNearestNeighborsAlgorithm();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        int k = 1;

        params.put(KNearestNeighborsAlgorithm.KEY_K, k);

        knnAlgorithm.setParams(params);

        System.out.println("propane dataSet learning curve (KNN with k=1):");
        ProjectUtils.printLearningCurve(propaneDataSet, knnAlgorithm);
    }
}
