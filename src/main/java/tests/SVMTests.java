package tests;

import algorithms.*;
import datasets.*;
import org.encog.ml.svm.*;
import org.encog.util.arrayutil.*;
import util.*;

import java.util.*;

public class SVMTests {

    public static void testKernels_IrisData() {
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

    public static void learningCurvePolynomialKernel_IrisData() {
        SvmAlgorithm svmAlgorithm = new SvmAlgorithm();
        Map<String, Object> params = new HashMap<>();

        params.put(SvmAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "species", 2, 0, 2, 0));
        params.put(SvmAlgorithm.KEY_KERNEL_TYPE, KernelType.Poly);
        params.put(SvmAlgorithm.KEY_C, 1.0);
        params.put(SvmAlgorithm.KEY_GAMMA, .25);

        svmAlgorithm.setParams(params);

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        System.out.println("iris dataSet learning curve (Polynomial Kernel):");
        ProjectUtils.printLearningCurve(irisDataSet, svmAlgorithm);
    }

    public static void crossValidateForCAndGamma_IrisData() {
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

    public static void learningCurveRBFKernel_IrisData() {
        SvmAlgorithm svmAlgorithm = new SvmAlgorithm();

        Map<String, Object> params = new HashMap<>();

        params.put(SvmAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "species", 2, 0, 2, 0));
        params.put(SvmAlgorithm.KEY_KERNEL_TYPE, KernelType.RadialBasisFunction);
        params.put(SvmAlgorithm.KEY_C, 0.28999999999999937);
        params.put(SvmAlgorithm.KEY_GAMMA, 0.35999999999999943);

        svmAlgorithm.setParams(params);

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        System.out.println("iris dataSet learning curve (SVM with RBF kernel C=0.29, gamma=0.36):");
        ProjectUtils.printLearningCurve(irisDataSet, svmAlgorithm);
    }

    private static double[] testSvmOnIrisData(KernelType kernelType, double C, double gamma) {
        SvmAlgorithm svmAlgorithm = new SvmAlgorithm();
        Map<String, Object> params = new HashMap<>();

        params.put(SvmAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "species", 2, 0, 2, 0));
        params.put(SvmAlgorithm.KEY_KERNEL_TYPE, kernelType);
        params.put(SvmAlgorithm.KEY_C, C);
        params.put(SvmAlgorithm.KEY_GAMMA, gamma);

        svmAlgorithm.setParams(params);

        DataSet<IrisInstance> irisDataSet = new IrisDataReader().getIrisDataSet();

        return ProjectUtils.leaveOneOutCrossValidate(irisDataSet, svmAlgorithm);
    }

    public static void testSeveralKernels_PropaneData() {

        double[] linearResults = testSvmOnPropaneData(KernelType.Linear, 1, .25);
        double[] polyResults = testSvmOnPropaneData(KernelType.Poly, 1, .25);
        double[] sigmoidResults = testSvmOnPropaneData(KernelType.Sigmoid, 1, .25);
        double[] rbfResults = testSvmOnPropaneData(KernelType.RadialBasisFunction, 1, 1.0000000000000003E-11);

        System.out.println("testing propane data with several kernels:");
        System.out.println("linear kernel err:" + linearResults[0]);
        System.out.println("poly kernel err:" + polyResults[0]);
        System.out.println("sigmoid kernel err:" + sigmoidResults[0]);
        System.out.println("rbf kernel err:" + rbfResults[0]);
    }

    public static void learningCurve_PropaneData() {
        SvmAlgorithm svmAlgorithm = new SvmAlgorithm();

        Map<String, Object> params = new HashMap<>();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        params.put(SvmAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "isLow", 1,0,1,-1));
        params.put(SvmAlgorithm.KEY_KERNEL_TYPE, KernelType.Linear);
        params.put(SvmAlgorithm.KEY_C, 1.0);

        params.put(SvmAlgorithm.KEY_GAMMA, 1.0000000000000003E-11);

        svmAlgorithm.setParams(params);

        System.out.println("iris dataSet learning curve (Linear Kernel SVM):");
        ProjectUtils.printLearningCurve(propaneDataSet, svmAlgorithm);
    }

    private static double[] testSvmOnPropaneData(KernelType kernelType, double C, double gamma) {
        SvmAlgorithm svmAlgorithm = new SvmAlgorithm();
        Map<String, Object> params = new HashMap<>();

        DataSet<PropaneInstance> propaneDataSet = new PropaneDataReader().getPropaneDataSet();

        params.put(SvmAlgorithm.KEY_OUTPUT_NORMALIZER, new NormalizedField(NormalizationAction.Normalize, "isLow", 1,0,1,-1));
        params.put(SvmAlgorithm.KEY_KERNEL_TYPE, kernelType);
        params.put(SvmAlgorithm.KEY_C, C);

        params.put(SvmAlgorithm.KEY_GAMMA, gamma);

        svmAlgorithm.setParams(params);

        return ProjectUtils.crossValidate(propaneDataSet, 10, svmAlgorithm);
    }
}
