package main;

import algorithms.*;
import tests.*;

/**
 * Created by brian on 5/26/16.
 */
public class Main {

    private static boolean shouldRunTest;
    private static boolean shouldRunGroup;
    private static boolean shouldPrintHelp;

    private static int integerArgument;

    public static void main(String[] args) {
        parseArgs(args);

        if (shouldPrintHelp) {
            printHelp();
        } else if (shouldRunTest) {
            runTest(integerArgument);
        } else if (shouldRunGroup) {
            runGroup(integerArgument);
        }
    }

    private static void parseArgs(String[] args) {
        String firstArg;
        if (args.length > 0) {
            firstArg = args[0].toLowerCase().trim();
        } else {
            shouldPrintHelp = true;
            return;
        }

        if (firstArg.equals("-t")) {
            shouldRunTest = true;
            parseNumberArg(args);

        } else if (firstArg.equals("-g")) {
            shouldRunGroup = true;
            parseNumberArg(args);

        } else {
            shouldPrintHelp = true;
        }
    }

    private static void parseNumberArg(String[] args) {
        if (args.length != 2) {
            shouldPrintHelp = true;
        }
        try {
            integerArgument = Integer.parseInt(args[1]);
        } catch (Exception e) {
            shouldPrintHelp = true;
        }
    }

    private static void printHelp() {
        System.out.println("Machine Learning experiments using Brian Murphy's Propane dataset and Fisher's Iris dataset.");
        System.out.println("use the argument -t [test number] to run a specific test.");
        System.out.println("use the argmuent -g [group number] to run a group of tests.");
    }

    private static void runTest(int testNumber) {
        switch (testNumber) {
            case 1:
                KNNTests.crossValidateForK_IrisData();
                break;
            case 2:
                KNNTests.learningCurve7NN_IrisData();
                break;
            case 3:
                KNNTests.crossValidateForK_PropaneData();
                break;
            case 4:
                KNNTests.learningCurve1NN_PropaneData();
                break;
            case 5:
                DecisionTreeTests.testC45_PropaneData();
                break;
            case 6:
                DecisionTreeTests.testC45AgressivePruning_PropaneData();
                break;
            case 7:
                DecisionTreeTests.testC45REP_PropaneData();
                break;
            case 8:
                DecisionTreeTests.testC45BinarySplits_PropaneData();
                break;
            case 9:
                DecisionTreeTests.testC45_IrisData();
                break;
            case 10:
                DecisionTreeTests.testC45AgressivePruning_IrisData();
                break;
            case 11:
                DecisionTreeTests.testC45REP_IrisData();
                break;
            case 12:
                DecisionTreeTests.testC45BinarySplits_IrisData();
                break;
            case 13:
                DecisionTreeTests.learningCurveC45_IrisData();
                break;
            case 14:
                DecisionTreeTests.learningCurveC45_PropaneData();
                break;
            case 15:
                BoostingTests.testBoostedC45_IrisData();
                break;
            case 16:
                BoostingTests.testBoostedDecisionStumps_IrisData();
                break;
            case 17:
                BoostingTests.learningCurveBoostedDecisionStumps_IrisData();
                break;
            case 18:
                BoostingTests.testBoostedDecisionStumps_PropaneData();
                break;
            case 19:
                BoostingTests.testBoostedC45_PropaneData();
                break;
            case 20:
                BoostingTests.learningCurveBoostedC45_PropaneData();
                break;
            case 21:
                NeuralNetTests.crossValidateErrorThreshold_IrisData();
                break;
            case 22:
                NeuralNetTests.crossValidateHiddenLayerSize_IrisData();
                break;
            case 23:
                NeuralNetTests.learningCurve_IrisData();
                break;
            case 25:
                NeuralNetTests.crossValidateErrorThreshold_PropaneData();
                break;
            case 26:
                NeuralNetTests.learningCurve_propaneData();
                break;
            case 27:
                SVMTests.learningCurvePolynomialKernel_IrisData();
                break;
            case 28:
                SVMTests.crossValidateForCAndGamma_IrisData();
                break;
            case 29:
                SVMTests.learningCurveRBFKernel_IrisData();
                break;
            case 30:
                SVMTests.testSeveralKernels_PropaneData();
                break;
            case 31:
                SVMTests.learningCurve_PropaneData();
                break;
            default:
                System.out.println("No such test");
                break;
        }
    }

    private static void runGroup(int groupNumber) {
        int[] tests;

        switch (groupNumber) {
            case 0:
                tests = new int[]{0, 1, 2};
                break;
            default:
                tests = new int[]{};
                break;
        }

        for (int testNumber : tests) {
            runTest(testNumber);
        }
    }
}
