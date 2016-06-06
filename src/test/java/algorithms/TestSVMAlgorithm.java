package algorithms;

import datasets.*;
import org.encog.ml.svm.*;
import org.encog.util.arrayutil.*;
import org.junit.*;
import util.*;

import java.util.*;

public class TestSvmAlgorithm {

    private SvmAlgorithm svmAlgorithm;

    @Before
    public void setUp() {
        svmAlgorithm = new SvmAlgorithm();
    }

    @Test
    public void testSvmOnPropaneData() {
        Map<String, Object> params = new HashMap<>();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        params.put(SvmAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "isLow", 1,-1,1,-1));
        params.put(SvmAlgorithm.KEY_KERNEL_TYPE, KernelType.Poly);
        params.put(SvmAlgorithm.KEY_C, 1.0);

        int inputLength = propaneDataSet.getInstances()[0].getInput().length;
        params.put(SvmAlgorithm.KEY_GAMMA, 1.0 / inputLength);

        svmAlgorithm.setParams(params);

        System.out.println("\npropane dataset: ");
        CrossValidation.crossValidate(propaneDataSet, 10, svmAlgorithm);
    }

    @Test
    public void testSvmOnIrisData() {
        Map<String, Object> params = new HashMap<>();

        params.put(SvmAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "species", 2, 0, 2, 0));
        params.put(SvmAlgorithm.KEY_KERNEL_TYPE, KernelType.Linear);
        params.put(SvmAlgorithm.KEY_C, 1.0);
        params.put(SvmAlgorithm.KEY_GAMMA, .25);

        svmAlgorithm.setParams(params);

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        System.out.println("\niris dataset: ");
        CrossValidation.crossValidate(irisDataSet, 10, svmAlgorithm);
    }
}
