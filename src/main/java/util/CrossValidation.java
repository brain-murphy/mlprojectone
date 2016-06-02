package util;

import algorithms.*;
import datasets.*;

import java.util.*;

public class CrossValidation {

    public static void main(String[] args) {

    }

    public static void crossValidate(DataSet dataSet, int numFolds, Algorithm algorithm) {
        List<Instance>[] groups = dataSet.splitDataSetRandomly(numFolds);

        double[] errors = new double[numFolds];

        for (int i = 0; i < numFolds; i++) {
            Instance[] testingData = groups[i].toArray(new Instance[groups[i].size()]);

            DataSet<Instance> trainingDataSet = new DataSet<Instance>(combineAllListsExceptOne(groups, i));

            algorithm.train(trainingDataSet, .003f, 5000);

            double sumOfError = 0;

            for (Instance testInstance : testingData) {
                double output = (double) algorithm.evaluate(testInstance.getInput());
                double expectedOutput = (double) testInstance.getOutput();
                sumOfError += (Math.abs(output - expectedOutput) / expectedOutput);
            }

            errors[i] = sumOfError / testingData.length;
        }

        System.out.println("Average Error " + ProjectUtils.mean(errors) + " over " + numFolds + " folds of size " + groups[0].size());

    }

    private static Instance[] combineAllListsExceptOne(List<Instance>[] lists, int listToLeaveOut) {
        Instance[] combinedArray = new Instance[totalCountOfAllLists(lists) - lists[listToLeaveOut].size()];

        int combinedArrayIndex = 0;

        for (int listIndex = 0; listIndex < lists.length; listIndex++) {
            if (listIndex == listToLeaveOut) {
                continue;
            }

            for (Instance instance : lists[listIndex]) {
                combinedArray[combinedArrayIndex] = instance;
                combinedArrayIndex += 1;
            }
        }

        return combinedArray;
    }

    private static int totalCountOfAllLists(List[] lists) {
        int total = 0;
        for (List list : lists) {
            total += list.size();
        }

        return total;
    }

    public static void leaveOneOutCrossValidate(DataSet dataSet, Algorithm algorithm) {
        crossValidate(dataSet, dataSet.getInstances().length, algorithm);
    }

}
