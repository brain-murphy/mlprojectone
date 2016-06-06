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
    public void testNeuralNetTrainingOnPropaneData() {
        DataSet<PropaneInstance> propaneData = new PropaneDataReader().getPropaneDataSet();

        Map<String, Object> params = new HashMap<>();
        params.put(NeuralNetAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "weight", 1, 0, 1, 0));
        params.put(NeuralNetAlgorithm.KEY_MAX_ITERATIONS, 500);
        params.put(NeuralNetAlgorithm.KEY_TARGET_ERROR, .02f);

        int inputArrayLength = propaneData.getInstances()[0].getInput().length;

        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, inputArrayLength));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 10));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        network.getStructure().finalizeStructure();
        network.reset();
        params.put(NeuralNetAlgorithm.KEY_NETWORK, network);


        neuralNetAlgorithm.setParams(params);

        System.out.println("\npropane data:");
        CrossValidation.crossValidate(propaneData, 10, neuralNetAlgorithm);
    }

    @Test
    public void testNeuralNetTrainingOnIrisData() {
        System.out.println("\niris data:");
        double[] results = runIrisTestWithParams(0.01f, 5000);
        runIrisTestWithParams(0.01f, 5000);
//        System.out.println("error: " + results[0] + " stdev: " + results[1]);
    }

//    @Test
//    public void searchNeuralNetTrainingOnIrisData() {
//        for (int errorDenominator = 10; errorDenominator < 1000; errorDenominator += 2) {
//            for (int maxIterations = 50; maxIterations < 5001; maxIterations *= 10) {
//                float trainingErrorThreshold = 1.0f / errorDenominator;
//                double[] results = runIrisTestWithParams(trainingErrorThreshold, maxIterations);
//                if (results[0] < .1 && results[1] < .1) {
//                    System.out.println("error: " + results[0] + " stdev: " + results[1] + " at training error threshold: " + trainingErrorThreshold + " and maxIterations: " + maxIterations);
//                }
//            }
//        }
//    }

    private double[] runIrisTestWithParams(float trainingError, int maxIterations) {
        DataSet<IrisInstance> irisData = new IrisDataReader().getIrisDataSet();

        Map<String, Object> params = new HashMap<>();

        params.put(NeuralNetAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "species", 2, 0, 1, 0));
        params.put(NeuralNetAlgorithm.KEY_TARGET_ERROR, trainingError);
        params.put(NeuralNetAlgorithm.KEY_MAX_ITERATIONS, maxIterations);

        BasicNetwork network = new BasicNetwork();

        int inputArrayLength = irisData.getInstances()[0].getInput().length;

        network.addLayer(new BasicLayer(null, true, inputArrayLength));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        network.getStructure().finalizeStructure();
        network.reset();
        params.put(NeuralNetAlgorithm.KEY_NETWORK, network);

        neuralNetAlgorithm.setParams(params);

//        System.out.println("\niris data:");
        return CrossValidation.crossValidate(irisData, 10, neuralNetAlgorithm);
    }
}
