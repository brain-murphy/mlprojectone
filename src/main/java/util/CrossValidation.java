package util;

import algorithms.*;
import datasets.*;
import org.apache.commons.math3.stat.descriptive.*;

import java.util.*;

public class CrossValidation {

    public static double[] crossValidate(DataSet dataSet, int numFolds, Algorithm algorithm) {
        List<Instance>[] groups = dataSet.splitDataSetRandomly(numFolds);

        SummaryStatistics errors = new SummaryStatistics();

        for (int i = 0; i < numFolds; i++) {
            Instance[] testingData = groups[i].toArray(new Instance[groups[i].size()]);

            DataSet<Instance> trainingDataSet = new DataSet<Instance>(combineAllListsExceptOne(groups, i));

            algorithm.train(trainingDataSet);

            double sumOfError = 0;

            for (Instance testInstance : testingData) {
                double output = (double) algorithm.evaluate(testInstance.getInput());
                sumOfError += testInstance.getError(output);
            }

            errors.addValue(sumOfError / testingData.length);
        }

        System.out.println("Average Error " + errors.getMean() + " over " + numFolds + " folds of size " + groups[0].size() + " with stdev " + errors.getStandardDeviation());
        return new double[] {errors.getMean(), errors.getStandardDeviation()};
    }

    private static Instance[] combineAllListsExceptOne(List<Instance>[] lists, int listToLeaveOut) {
        if (lists.length == 1) {
            return lists[0].toArray(new Instance[lists[0].size()]);
        }

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

    public static double[] leaveOneOutCrossValidate(DataSet dataSet, Algorithm algorithm) {
        return crossValidate(dataSet, dataSet.getInstances().length, algorithm);
    }

}
