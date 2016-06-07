package tests;

import algorithms.*;
import com.sun.org.apache.bcel.internal.generic.*;
import datasets.*;
import org.encog.engine.network.activation.*;
import org.encog.neural.networks.*;
import org.encog.neural.networks.layers.*;
import org.encog.util.arrayutil.*;
import util.*;

import java.util.*;

public class NeuralNetTests {
    public static void crossValidateErrorThreshold_IrisData() {
        System.out.println("\ncross validating iris data for error threshold:");

        int start = 2;
        int end = 1025;

        for (int thresholdDivisor = start; thresholdDivisor < end; thresholdDivisor <<= 1) {
            float errorThreshold = 1.0f / thresholdDivisor;
            double error = runIrisTestWithParams(errorThreshold, 59, 16)[0];

            System.out.println("error threshold:" + errorThreshold + " error:" + error);
        }
    }

    public static void crossValidateHiddenLayerSize_IrisData() {
        System.out.println("\ncross validating iris data for hidden layer size:");

        int start = 2;
        int end = 80;

        double[] errors = new double[end - start];

        for (int layerLength = start; layerLength < end; layerLength++) {
            errors[layerLength - start] = runIrisTestWithParams(0.0078125f, 59, layerLength)[0];

            System.out.println("hidden layer length:" + layerLength + " error:" + errors[layerLength - start]);
        }
    }

    public static void learningCurve_IrisData() {
        System.out.println("\niris dataSet learning curve (NeuralNet with sigmoid activation and 13 node hidden layer):");
        System.out.println("trainingIterations,trainingError,crossValidationError");

        for (int iterations = 2; iterations < 100; iterations++) {

            double[] results = runIrisTestWithParams(0.0078125f, iterations, 13);

            System.out.println(Integer.toString(iterations) + "," + results[4] + "," + String.format("%.3f", results[0]));
        }
    }

    private static double[] runIrisTestWithParams(float trainingError, int maxIterations, int layerLength) {
        NeuralNetAlgorithm neuralNetAlgorithm = new NeuralNetAlgorithm();

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

        return ProjectUtils.leaveOneOutCrossValidate(dataSet, neuralNetAlgorithm);
    }

    public static void crossValidateErrorThreshold_PropaneData() {
        System.out.println("\ncross validating propane data for error threshold:");

        int start = 2;
        int end = 1025;

        for (int thresholdDivisor = start; thresholdDivisor < end; thresholdDivisor <<= 1) {
            float errorThreshold = 1.0f / thresholdDivisor;
            double error = runPropaneTestWithParams(errorThreshold, 5000, 10)[0];

            System.out.println("error threshold:" + errorThreshold + " error:" + error);
        }
    }

    public static void learningCurve_propaneData() {
        System.out.println("\npropane dataSet learning curve (NeuralNet with sigmoid activation and 12 node hidden layer):");
        System.out.println("trainingIterations,trainingError,crossValidationError");

        for (int iterations = 2; iterations < 100; iterations++) {

            double[] results = runPropaneTestWithParams(0.01f, iterations, 12);

            System.out.println(Integer.toString(iterations) + "," + results[4] + "," + String.format("%.3f", results[0]));
        }
    }


    private static double[] runPropaneTestWithParams(float trainingError, int maxIterations, int layerLength) {
        NeuralNetAlgorithm neuralNetAlgorithm = new NeuralNetAlgorithm();
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

        return ProjectUtils.crossValidate(dataSet, 10, neuralNetAlgorithm);
    }

}
