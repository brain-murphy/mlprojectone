package algorithms;

import datasets.*;
import org.junit.*;
import util.*;

public class TestNeuralNetAlgorithm {

    private NeuralNetAlgorithm neuralNetAlgorithm;

    @Before
    public void setUp() {
        neuralNetAlgorithm = new NeuralNetAlgorithm();
    }

    @Test
    public void testNeuralNetTraining() {
        DataSet<PropaneInstance> dataSet = new PropaneDataReader().getPropaneDataset();
        CrossValidation.crossValidate(dataSet, 60, neuralNetAlgorithm);
    }
}
