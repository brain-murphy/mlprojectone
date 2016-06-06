package algorithms;

import datasets.*;
import org.junit.*;
import util.*;

import java.util.*;

public class TestDecisionTreeAlgorithm {

    private DecisionTreeAlgorithm decisionTreeAlgorithm;

    @Before
    public void setUp() {
        decisionTreeAlgorithm = new DecisionTreeAlgorithm();
    }

    @Test
    public void testPropaneData() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, false);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\npropane dataset (C4.5 tree):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(propaneDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testPropaneDataReducedErrorPruning() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, true);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\npropane dataset (C4.5 with reduced error pruning):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(propaneDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testPropaneDataPruningConfidence() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        double pruningConfidence = 0.1;

        params.put(DecisionTreeAlgorithm.KEY_PRUNING_CONFIDENCE, pruningConfidence);

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, false);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\npropane dataset (C4.5 with aggressive pruning):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(propaneDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testPropaneDataBinarySplitsOnly() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, false);

        params.put(DecisionTreeAlgorithm.KEY_ONLY_BINARY_SPLITS, true);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\npropane dataset (C4.5 with only binary splits):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(propaneDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }


    @Test
    public void testIrisDataC45() {
        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, false);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\niris datset (C4.5 tree):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(irisDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testIrisDataRep() {
        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(DecisionTreeAlgorithm.KEY_REDUCED_ERROR_PRUNING, true);

        decisionTreeAlgorithm.setParams(params);

        System.out.println("\niris datset (C4.5 with reduced error pruning):");
        double[] results = CrossValidation.leaveOneOutCrossValidate(irisDataSet, decisionTreeAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

}
