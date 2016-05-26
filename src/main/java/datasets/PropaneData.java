package datasets;


import util.*;

import java.io.*;
import java.util.*;


public class PropaneData {

    private static final String PROPANE_DATA_FILE_PATH = "datasets/propaneData.ser";

    private Map<Float,List<Map<Integer,Integer>>> data;
    private float[] weights;

    public static PropaneData readPropaneDataFromResources() {
        PropaneData propaneData = new PropaneData(deserializePropaneData());

        return propaneData;
    }

    private static Map<Float,List<Map<Integer,Integer>>> deserializePropaneData() {
        Map<Float,List<Map<Integer,Integer>>> data = null;
        try {
            FileInputStream fileIn = new FileInputStream(PROPANE_DATA_FILE_PATH);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            data = (Map<Float,List<Map<Integer,Integer>>>) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        return data;
    }

    private PropaneData(Map<Float,List<Map<Integer,Integer>>> pData) {
        data = pData;

        weights = ProjectUtils.toPrimitiveFloatArray(data.keySet());
    }

    public List<Map<Integer,Integer>> getFftsForTankWeight(float weight) {
        return data.get(weight);
    }

    public float[] getWeights() {
        return weights;
    }

    public static void main(String args[]) {
        PropaneData parser = new PropaneData(null);

        String[] fileNames = parser.getFileNames();
        float[] weights = parser.parseWeights(fileNames);

        String[] fileContents = parser.getFileContents(fileNames);

        Map<Float, List<Map<Integer, Integer>>> data = new HashMap<>();

        for (int i = 0; i < fileNames.length; i++) {
            data.put(weights[i], parser.parseFileContents(fileContents[i]));
        }

        parser.serializeData(data);
    }

    private void serializeData(Object data) {
        try {
            FileOutputStream outputStream = new FileOutputStream("propaneData.ser");
            ObjectOutputStream out = new ObjectOutputStream(outputStream);
            out.writeObject(data);
            out.close();
            outputStream.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private List<Map<Integer,Integer>> parseFileContents(String fileContents) {
        String[] lines = fileContents.split(System.getProperty("line.separator"));

        List<Map<Integer, Integer>> fftsForFile = new LinkedList<>();

        Map<Integer, Integer> currentFft = new HashMap<>();

        for (String line : lines) {
            if (isLineValid(line)) {
                String[] splitLine = line.split(" ");

                int frequency = Integer.parseInt(splitLine[0]);
                int magnitude = Integer.parseInt(splitLine[1]);

                currentFft.put(frequency, magnitude);
            } else {
                fftsForFile.add(currentFft);

                currentFft = new HashMap<>();
            }
        }

        return fftsForFile;
    }

    private boolean isLineValid(String line) {
        String[] splitLine = line.split(" ");

        final int radix = 10;

        if (splitLine.length > 1 && isInteger(splitLine[0], radix) && isInteger(splitLine[1], radix)) {
            return true;
        }

        return false;
    }

    private String[] getFileContents(String[] fileNames) {
        String[] fileContents = new String[fileNames.length];

        for (int i = 0; i < fileNames.length; i++) {
            fileContents[i] = getFile("datasets/propaneData/" + fileNames[i]);
        }

        return fileContents;
    }

    private float[] parseWeights(String[] fileNames) {
        float[] weights = new float[fileNames.length];

        for (int i = 0; i < fileNames.length; i++) {
            weights[i] = Float.parseFloat(removeFileExtension(fileNames[i]));
        }

        return weights;
    }

    public String removeFileExtension(String fileName) {
        int extensionStartIndex = fileName.indexOf(".txt");
        return fileName.substring(0, extensionStartIndex);
    }

    private String getFile(String fileName) {

        StringBuilder result = new StringBuilder("");

        //Get file from resources folder
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource(fileName).getFile());

        try (Scanner scanner = new Scanner(file)) {

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                result.append(line).append("\n");
            }

            scanner.close();

        } catch (IOException e) {
            e.printStackTrace();
        }

        return result.toString();

    }

    private String[] getFileNames() {
        return new String[]{
                "18.1.txt",                "22.001.txt",                "25.2.txt",                "29.5001.txt",                "33.5.txt",
                "18.8.txt",                "22.75.txt",                "25.21.txt",                "29.9.txt",                "34.15.txt",
                "18.9.txt",                "22.txt",                "26.6.txt",                "30.35.txt",                "34.2.txt",
                "19.1.txt",                "23.5.txt",                "26.65.txt",                "31.4.txt",                "34.5.txt",
                "19.2.txt",                "23.5001.txt",                "27.4.txt",                "31.65.txt",                "34.6.txt",
                "19.5.txt",                "23.75.txt",                "27.5.txt",                "32.55.txt",                "34.8.txt",
                "19.85.txt",                "23.8.txt",                "27.txt",                "32.65.txt",                "36.6.txt",
                "20.1.txt",                "24.1.txt",                "28.3.txt",                "32.7.txt",
                "20.75.txt",                "24.4.txt",                "28.45.txt",                "32.9.txt",
                "21.5.txt",                "24.7.txt",                "29.5.txt",                "33.4.txt",        };
    }

    public static boolean isInteger(String s, int radix) {
        if(s.isEmpty()) return false;
        for(int i = 0; i < s.length(); i++) {
            if(i == 0 && s.charAt(i) == '-') {
                if(s.length() == 1) return false;
                else continue;
            }
            if(Character.digit(s.charAt(i),radix) < 0) return false;
        }
        return true;
    }

}
