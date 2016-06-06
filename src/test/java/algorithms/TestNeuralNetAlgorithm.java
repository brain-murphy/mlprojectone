package algorithms;

import datasets.*;
import org.encog.engine.network.activation.*;
import org.encog.neural.networks.*;
import org.encog.neural.networks.layers.*;
import org.encog.util.arrayutil.*;
import org.junit.*;
import util.*;

import java.util.*;

public class TestNeuralNetAlgorithm {

    private NeuralNetAlgorithm neuralNetAlgorithm;

    @Before
    public void setUp() {
        neuralNetAlgorithm = new NeuralNetAlgorithm();
    }

    @Test
    public void crossValidateForLayerSizeIris() {
        System.out.println("\ncross validating iris data for hidden layer size:");

        int start = 2;
        int end = 80;

        double[] errors = new double[end - start];

        for (int layerLength = start; layerLength < end; layerLength++) {
            errors[layerLength - start] = runIrisTestWithParams(0.00390625f, 50, layerLength)[0];

            System.out.println("hidden layer length:" + layerLength + " error:" + errors[layerLength - start]);
        }
    }

    @Test
    public void crossValidateForErrorThresholdIris() {
        System.out.println("\ncross validating iris data for error threshold:");

        int start = 2;
        int end = 1025;

        for (int thresholdDivisor = start; thresholdDivisor < end; thresholdDivisor <<= 1) {
            float errorThreshold = 1.0f / thresholdDivisor;
            double error = runIrisTestWithParams(errorThreshold, 50, 5)[0];

            System.out.println("error threshold:" + errorThreshold + " error:" + error);
        }
    }

    @Test
    public void crossValidateForMaxIterationsIris() {
        System.out.println("\ncross validating iris data for max iterations:");

        int start = 10;
        int end = 100;

        for (int maxIterations = start; maxIterations < end; maxIterations += 1) {
            double error = runIrisTestWithParams(0.00390625f, maxIterations, 5)[0];

            System.out.println("max iterations:" + maxIterations + " error:" + error);
        }
    }

    @Test
    public void crossValidateForLayerSizePropane() {
        System.out.println("\ncross validating propane data for hidden layer size:");

        int start = 5;
        int end = 200;

        double[] errors = new double[end - start];

        for (int layerLength = start; layerLength < end; layerLength++) {
            errors[layerLength - start] = runPropaneTestWithParams(0.01f, 300, layerLength)[0];

            System.out.println("hidden layer length:" + layerLength + " error:" + errors[layerLength - start]);
        }
    }

    @Test
    public void crossValidateForErrorThresholdPropane() {
        System.out.println("\ncross validating propane data for error threshold:");

        int start = 2;
        int end = 1025;

        for (int thresholdDivisor = start; thresholdDivisor < end; thresholdDivisor <<= 1) {
            float errorThreshold = 1.0f / thresholdDivisor;
            double error = runPropaneTestWithParams(errorThreshold, 5000, 10)[0];

            System.out.println("error threshold:" + errorThreshold + " error:" + error);
        }
    }

    @Test
    public void crossValidateForMaxIterationsPropane() {
        System.out.println("\ncross validating propane data for max iterations:");

        int start = 1000;
        int end = 10001;

        for (int maxIterations = start; maxIterations < end; maxIterations += 1000) {
            double error = runPropaneTestWithParams(0.01f, maxIterations, 10)[0];

            System.out.println("max iterations:" + maxIterations + " error:" + error);
        }
    }

    @Test
    public void testThreeLayersPropaneData() {
        System.out.println("\ncross validating propane data with a three layer net");

        DataSet<PropaneInstance> dataSet = new PropaneDataReader().getPropaneDataSet();
        Map<String, Object> params = new HashMap<>();

        params.put(NeuralNetAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "isLow", 1, 0, 1, 0));
        params.put(NeuralNetAlgorithm.KEY_TARGET_ERROR, .01f);
        params.put(NeuralNetAlgorithm.KEY_MAX_ITERATIONS, 500);

        BasicNetwork network = new BasicNetwork();

        int inputArrayLength = dataSet.getInstances()[0].getInput().length;

        network.addLayer(new BasicLayer(null, true, inputArrayLength));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 11));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 10));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        network.getStructure().finalizeStructure();
        network.reset();
        params.put(NeuralNetAlgorithm.KEY_NETWORK, network);

        neuralNetAlgorithm.setParams(params);

        double[] results = CrossValidation.leaveOneOutCrossValidate(dataSet, neuralNetAlgorithm);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testAllBestParamsIrisData() {
        float bestErrorThreshold = 0.00390625f;
        int bestMaxIterations = 59;
        int bestHiddenLayerSize = 16;

        double[] results = runIrisTestWithParams(bestErrorThreshold, bestMaxIterations, bestHiddenLayerSize);

        System.out.println("\nLOOCV for best iris data parameter combination:");
        System.out.println("training error stopping threshold:" + bestErrorThreshold + "\nmax iterations:" + bestMaxIterations + "\nnumber hidden layer nodes:" + bestHiddenLayerSize);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    @Test
    public void testAllBestParamsPropaneData() {
        double[] results = runPropaneTestWithParams(0.00390625f, 59, 16);

        System.out.println("Error:" + results[0] + " over " + ((int) results[2]) + " folds of size " + ((int)results[3]) + " with stdev " + results[1]);
    }

    private double[] runIrisTestWithParams(float trainingError, int maxIterations, int layerLength) {
        DataSet<IrisInstance> dataSet = new IrisDataReader().getIrisDataSet();
        Map<String, Object> params = new HashMap<>();

        params.put(NeuralNetAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "species", 2, 0, 1, 0));
        params.put(NeuralNetAlgorithm.KEY_TARGET_ERROR, trainingError);
        params.put(NeuralNetAlgorithm.KEY_MAX_ITERATIONS, maxIterations);

        BasicNetwork network = new BasicNetwork();

        int inputArrayLength = dataSet.getInstances()[0].getInput().length;

        network.addLayer(new BasicLayer(null, true, inputArrayLength));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, layerLength));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        network.getStructure().finalizeStructure();
        network.reset();
        params.put(NeuralNetAlgorithm.KEY_NETWORK, network);

        neuralNetAlgorithm.setParams(params);

        return CrossValidation.leaveOneOutCrossValidate(dataSet, neuralNetAlgorithm);
    }

    private double[] runPropaneTestWithParams(float trainingError, int maxIterations, int layerLength) {
        DataSet<PropaneInstance> dataSet = new PropaneDataReader().getPropaneDataSet();
        Map<String, Object> params = new HashMap<>();

        params.put(NeuralNetAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "isLow", 1, 0, 1, 0));
        params.put(NeuralNetAlgorithm.KEY_TARGET_ERROR, trainingError);
        params.put(NeuralNetAlgorithm.KEY_MAX_ITERATIONS, maxIterations);

        BasicNetwork network = new BasicNetwork();

        int inputArrayLength = dataSet.getInstances()[0].getInput().length;

        network.addLayer(new BasicLayer(null, true, inputArrayLength));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, layerLength));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        network.getStructure().finalizeStructure();
        network.reset();
        params.put(NeuralNetAlgorithm.KEY_NETWORK, network);

        neuralNetAlgorithm.setParams(params);

        return CrossValidation.crossValidate(dataSet, 10, neuralNetAlgorithm);
    }
}
