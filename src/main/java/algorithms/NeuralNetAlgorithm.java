package algorithms;

import com.sun.deploy.util.*;
import datasets.*;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.basic.*;
import org.encog.neural.networks.*;
import org.encog.neural.networks.layers.*;
import org.encog.neural.networks.training.propagation.resilient.*;
import org.encog.util.arrayutil.*;

import java.util.*;

public class NeuralNetAlgorithm implements Algorithm {

    private static final long LOGGING_INTERVAL = 1000; // ms

    private BasicNetwork network;
    private double[][] input;
    private double[][] output;
    private int epoch;
    private double error;

    private Timer loggingTimer;
    private NormalizedField normalizeOutput;


    @Override
    public void train(DataSet pDataSet, float targetError, int iterations) {
        parseTrainingData(pDataSet);
        createNetwork();

        BasicMLDataSet dataSet = new BasicMLDataSet(input, output);

        ResilientPropagation trainer = new ResilientPropagation(network, dataSet);

        startTimedLogging();

        epoch = 1;
        do {
            trainer.iteration();
            error = trainer.getError();
            epoch++;
        } while (error > targetError && epoch < iterations);

        loggingTimer.cancel();
    }

    private void startTimedLogging() {
        TimerTask loggingTimerTask = new TimerTask() {
            @Override
            public void run() {
                System.out.println("Error " + error + " at iteration " + epoch);
            }
        };
        loggingTimer = new Timer();
        loggingTimer.schedule(loggingTimerTask, LOGGING_INTERVAL, LOGGING_INTERVAL);
    }

    private void createNetwork() {
        network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, input[0].length));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 10));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 10));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        network.getStructure().finalizeStructure();
        network.reset();
    }

    private void parseTrainingData(DataSet dataSet) {
        Instance[] instances = dataSet.getInstances();
        input = new double[instances.length][];
        output = new double[instances.length][];

        NormalizeArray normalizer = new NormalizeArray();
        normalizer.setNormalizedLow(0);

        normalizeOutput = new NormalizedField(NormalizationAction.Normalize, "weight", 37, 0, 1, 0);

        for (int i = 0; i < instances.length; i++) {
            input[i] = normalizer.process((double[]) instances[i].getInput());
            output[i] = new double[] {normalizeOutput.normalize((double) (instances[i].getOutput()))};
        }
    }

    @Override
    public Object evaluate(Object input) {
        return normalizeOutput.deNormalize(network.compute(new BasicMLData((double[]) input)).getData()[0]);
    }
}
