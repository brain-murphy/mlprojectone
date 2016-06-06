package datasets;

import org.junit.*;
import static org.junit.Assert.*;

public class TestIrisDataReader {

    private IrisDataReader reader;

    @Before
    public void setUp() {
        reader = new IrisDataReader();
    }

    @Test
    public void testDataSetIsValid() {
        DataSet<IrisInstance> dataSet = reader.getIrisDataSet();

        for (IrisInstance i : dataSet) {
            assertTrue("instance is not valid", instanceIsValid(i));
        }
    }

    @Test
    public void testIrisErrorFunction() {
        IrisInstance setosa = new IrisInstance(0,0,0,0,"Iris-setosa");
        assertTrue("Not right", setosa.getError(0) == 0);
        assertTrue(setosa.getError(.99) == 1);
        IrisInstance versicolor = new IrisInstance(0,0,0,0,"Iris-versicolor");
        assertTrue(versicolor.getError(1) == 0);
        IrisInstance virginica = new IrisInstance(0,0,0,0,"Iris-virginica");
        assertTrue(virginica.getError(2) == 0);
    }

    private boolean instanceIsValid(IrisInstance irisInstance) {
        for (double inputParameter : irisInstance.getInput()) {
            if (inputParameter <= 0) {
                return false;
            }
        }

        return true;
    }
}
