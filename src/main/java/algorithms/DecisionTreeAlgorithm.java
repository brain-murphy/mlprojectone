package algorithms;

import datasets.*;
import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.core.*;
import weka.core.Instance;

import java.util.*;

public class DecisionTreeAlgorithm implements Algorithm {



    private WekaParser parser;
    private Classifier tree;


    @Override
    public void setParams(Map<String, Object> params) {

    }

    @Override
    public void train(DataSet dataset) {
        parser = new WekaParser(dataset);

        tree = new REPTree();

        try {
            tree.buildClassifier(parser.getDataSetAsInstances());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public Object evaluate(Object input) {
        Instance wekaInstance = parser.parseInstanceForEvaluation((double[]) input);

        try {
            return tree.classifyInstance(wekaInstance);
        } catch (Exception e) {
            e.printStackTrace();
        }
        throw new RuntimeException("could not classify");
    }
}
