package datasets;

import org.junit.*;

/**
 * Created by brian on 5/26/16.
 */
public class TestPropaneData {

    PropaneData parser;

    @Before
    public void setUp() {
//        parser = new PropaneData();
    }

    @Test
    public void testRemoveFileExtensionFromName() {
        String fileNameWithExtension = "36.6.txt";
        String fileNameWithoutExtension = "36.6";

        String result = parser.removeFileExtension(fileNameWithExtension);

        assert(result.equals(fileNameWithoutExtension));
    }
}
