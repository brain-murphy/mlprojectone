package algorithms;

import datasets.*;
import weka.classifiers.*;
import weka.classifiers.meta.*;
import weka.core.*;

import java.util.*;

public class BoostingAlgorithm implements Algorithm {

    public static final String KEY_ALGORITHM_CLASS_NAME = "algorithm class param";

    private WekaParser parser;
    private Classifier booster;
    private String[] options;

    @Override
    public void setParams(Map<String, Object> params) {
        String algorithmClassName = (String) params.get(KEY_ALGORITHM_CLASS_NAME);
        try {
            options = Utils.splitOptions("-W " + algorithmClassName);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void train(DataSet dataset) {
        parser = new WekaParser(dataset);

        booster = new Bagging();

        try {
            booster.setOptions(options);
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            booster.buildClassifier(parser.getDataSetAsInstances());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    @Override
    public Object evaluate(Object input) {
        try {
            return booster.classifyInstance(parser.parseInstanceForEvaluation((double []) input));
        } catch (Exception e) {
            e.printStackTrace();
        }
        throw new RuntimeException("Cannot classify");
    }
}
