package algorithms;

import datasets.*;
import org.junit.*;
import util.*;

public class TestDecisionTreeAlgorithm {

    private DecisionTreeAlgorithm decisionTreeAlgorithm;

    @Before
    public void setUp() {
        decisionTreeAlgorithm = new DecisionTreeAlgorithm();
    }

    @Test
    public void testDecisionTreeOnPropaneData() {
        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        System.out.println("\npropane dataset:");
        CrossValidation.crossValidate(propaneDataSet, 30, decisionTreeAlgorithm);
    }

    @Test
    public void testDecisionTreeOnIrisData() {
        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        System.out.println("\niris datset:");
        CrossValidation.crossValidate(irisDataSet, 10, decisionTreeAlgorithm);
    }

}
