package algorithms;

import datasets.*;

public interface Algorithm {
    public void train(DataSet dataset, float targetTrainingError, int iterations);
    public Object evaluate(Object input);
}
