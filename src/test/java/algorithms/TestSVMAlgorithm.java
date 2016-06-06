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
    public void crossValidateForBestCIrisData() {

        OrderOfMagnitudeIterator ordersOfMagnitude = new OrderOfMagnitudeIterator();

        while (ordersOfMagnitude.hasNext()) {
            double C = ordersOfMagnitude.next();
            double[] results = testSvmOnIrisData(KernelType.Linear, C, .25);

            System.out.println("with C=" + C + " error:" + results[0]);
        }
    }

    @Test
    public void crossValidateForGammaIrisData() {
        OrderOfMagnitudeIterator ordersOfMagnitude = new OrderOfMagnitudeIterator();

        while (ordersOfMagnitude.hasNext()) {
            double gamma = ordersOfMagnitude.next();
            double[] results = testSvmOnIrisData(KernelType.RadialBasisFunction, 1, gamma);

            System.out.println("with gamma=" + gamma + " error:" + results[0]);
        }
    }

    @Test
    public void crossValidateForBestCPropaneData() {
        OrderOfMagnitudeIterator ordersOfMagnitude = new OrderOfMagnitudeIterator();

        while (ordersOfMagnitude.hasNext()) {
            double C = ordersOfMagnitude.next();
            double[] results = testSvmOnPropaneData(KernelType.RadialBasisFunction, C, 1E-11);

            System.out.println("with C=" + C + " error:" + results[0]);
        }
    }

    @Test
    public void crossValidateForGammaPropaneData() {
        OrderOfMagnitudeIterator ordersOfMagnitude = new OrderOfMagnitudeIterator();

        while (ordersOfMagnitude.hasNext()) {
            double gamma = ordersOfMagnitude.next();
            double[] results = testSvmOnPropaneData(KernelType.RadialBasisFunction, 1, gamma);

            System.out.println("with gamma=" + gamma + " error:" + results[0]);
        }
    }

    @Test
    public void testKernelsPropaneData() {
        double[] linearResults = testSvmOnPropaneData(KernelType.Linear, 1, .25);
        double[] polyResults = testSvmOnPropaneData(KernelType.Poly, 1, .25);
        double[] sigmoidResults = testSvmOnPropaneData(KernelType.Sigmoid, 1, .25);
        double[] rbfResults = testSvmOnPropaneData(KernelType.RadialBasisFunction, 1, .25);

        System.out.println("testing propane data with several kernels:");
        System.out.println("linear kernel err:" + linearResults[0]);
        System.out.println("poly kernel err:" + polyResults[0]);
        System.out.println("sigmoid kernel err:" + sigmoidResults[0]);
        System.out.println("rbf kernel err:" + rbfResults[0]);
    }

    @Test
    public void testKernelsIrisData() {
        double[] linearResults = testSvmOnIrisData(KernelType.Linear, 1, .25);
        double[] polyResults = testSvmOnIrisData(KernelType.Poly, 1, .25);
        double[] sigmoidResults = testSvmOnIrisData(KernelType.Sigmoid, 1, .25);
        double[] rbfResults = testSvmOnIrisData(KernelType.RadialBasisFunction, 0.28999999999999937, 0.35999999999999943);

        System.out.println("testing iris data with several kernels:");
        System.out.println("linear kernel err:" + linearResults[0]);
        System.out.println("poly kernel err:" + polyResults[0]);
        System.out.println("sigmoid kernel err:" + sigmoidResults[0]);
        System.out.println("rbf kernel err:" + rbfResults[0]);
    }

    @Test
    public void crossValidateIrisParamsInSmallRange() {
        double one_thousandth = .001;
        double one_hundredth = .01;

        double bestError = 1;

        for (double gamma = 1; gamma > one_thousandth; gamma -= one_hundredth) {
            for (double C = 1; C > one_thousandth; C -= one_hundredth) {
                double[] results = testSvmOnIrisData(KernelType.RadialBasisFunction, C, gamma);
                if (results[0] < bestError) {
                    bestError = results[0];
                    System.out.println("with gamma=" + gamma + " C=" + C + " error:" + results[0]);
                }
            }
        }
    }

    private double[] testSvmOnPropaneData(KernelType kernelType, double C, double gamma) {
        Map<String, Object> params = new HashMap<>();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        params.put(SvmAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "isLow", 1,0,1,-1));
        params.put(SvmAlgorithm.KEY_KERNEL_TYPE, kernelType);
        params.put(SvmAlgorithm.KEY_C, C);

        params.put(SvmAlgorithm.KEY_GAMMA, gamma);

        svmAlgorithm.setParams(params);

        return CrossValidation.crossValidate(propaneDataSet, 10, svmAlgorithm);
    }

    private double[] testSvmOnIrisData(KernelType kernelType, double C, double gamma) {
        Map<String, Object> params = new HashMap<>();

        params.put(SvmAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "species", 2, 0, 2, 0));
        params.put(SvmAlgorithm.KEY_KERNEL_TYPE, kernelType);
        params.put(SvmAlgorithm.KEY_C, C);
        params.put(SvmAlgorithm.KEY_GAMMA, gamma);

        svmAlgorithm.setParams(params);

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        return CrossValidation.leaveOneOutCrossValidate(irisDataSet, svmAlgorithm);
    }

    private class OrderOfMagnitudeIterator implements Iterator<Double> {

        private double current = 1E13;

        @Override
        public boolean hasNext() {
            return current > 1E-12;
        }

        @Override
        public Double next() {
            double last = current;
            current /= 10;
            return last;
        }
    }
}
