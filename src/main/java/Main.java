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
            case 0:
                ////
                break;
            default:
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
