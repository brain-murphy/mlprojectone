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
    public void testPropaneData() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        int k = 1;

        params.put(KNearestNeighborsAlgorithm.KEY_K, k);

        knnAlgorithm.setParams(params);

        double[] results = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, knnAlgorithm);

        System.out.println("Lowest Error:" + results[0] + " at k=" + k + " over " + ((int)results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);

    }

    @Test
    public void testIrisData() {
        DataSet<IrisInstance> irisInstanceDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        int k = 7;

        params.put(KNearestNeighborsAlgorithm.KEY_K, k);

        knnAlgorithm.setParams(params);

        double[] results = ProjectUtils.leaveOneOutCrossValidate(irisInstanceDataSet, knnAlgorithm);

        System.out.println("Lowest Error:" + results[0] + " at k=" + k + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void crossValidateForKPropaneData() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        int highestKToTest = 20;

        double[] errors = new double[highestKToTest];

        for (int k = 1; k <= highestKToTest; k++) {

            knnAlgorithm = new KNearestNeighborsAlgorithm();

            params.put(KNearestNeighborsAlgorithm.KEY_K, k);

            knnAlgorithm.setParams(params);

            errors[k - 1] = ProjectUtils.leaveOneOutCrossValidate(propaneDataSet, knnAlgorithm)[0];

            System.out.println("k=" + k + " error:" + errors[k - 1]);
        }
    }

    @Test
    public void crossValidateForKIrisData() {
        DataSet<IrisInstance> irisInstanceDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        int highestKToTest = 20;

        double[] errors = new double[highestKToTest];

        System.out.println("\niris data LOOCV for best k:");
        for (int k = 1; k <= highestKToTest; k++) {

            knnAlgorithm = new KNearestNeighborsAlgorithm();

            params.put(KNearestNeighborsAlgorithm.KEY_K, k);

            knnAlgorithm.setParams(params);

            errors[k - 1] = ProjectUtils.leaveOneOutCrossValidate(irisInstanceDataSet, knnAlgorithm)[0];

            System.out.println("k=" + k + " error:" + errors[k - 1]);
        }
    }

    @Test
    public void learningCurvePropaneData() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        int k = 1;

        params.put(KNearestNeighborsAlgorithm.KEY_K, k);

        knnAlgorithm.setParams(params);

        System.out.println("propane dataSet learning curve (KNN with k=1):");
        ProjectUtils.printLearningCurve(propaneDataSet, knnAlgorithm);
    }

    @Test
    public void learningCurveIrisData() {
        DataSet<IrisInstance> irisInstanceDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        int k = 7;

        params.put(KNearestNeighborsAlgorithm.KEY_K, k);

        knnAlgorithm.setParams(params);

        System.out.println("iris dataSet learning curve (KNN with k=7):");
        ProjectUtils.printLearningCurve(irisInstanceDataSet, knnAlgorithm);
    }
}
