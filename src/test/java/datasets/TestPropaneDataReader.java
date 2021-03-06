package datasets;

import org.junit.*;

import java.util.*;

import static org.junit.Assert.*;

/**
 * Created by brian on 5/26/16.
 */
public class TestPropaneDataReader {

    PropaneDataReader propaneDataReader;

    @Before
    public void setUp() {
        propaneDataReader = new PropaneDataReader();
    }

    @Test
    public void testPropaneDataErrorFunction() {
        Instance fullPropaneInstance = new PropaneInstance(new double[]{}, 1);
        Instance emptyPropaneInstance = new PropaneInstance(new double[]{}, -1);

        assertTrue("full Instance should return err=1", fullPropaneInstance.getError(0) == 1);
        assertTrue("full Instance should return err=1", fullPropaneInstance.getError(-1) == 1);
        assertTrue("full Instance should return err=0", fullPropaneInstance.getError(1) == 0);

        assertTrue("empty Instance should return err=1", emptyPropaneInstance.getError(1) == 1);
        assertTrue("empty Instance should return err=1", emptyPropaneInstance.getError(.1) == 1);
        assertTrue("empty Instance should return err=0", emptyPropaneInstance.getError(-0.9) == 0);
    }

    @Test
    public void testIterator() {
        //TODO
    }

    private int[] getAllFrequencies() {
        return new int[]{808,812,816,820,824,828,832,835,839,843,847,851,855,859,863,867,871,875,878,882,886,890,894,
                898,902,906,910,914,917,921,925,929,933,937,941,945,949,953,957,960,964,968,972,976,980,984,988,992,996,
                1000,1003,1007,1011,1015,1019,1023,1027,1031,1035,1039,1042,1046,1050,1054,1058,1062,1066,1070,1074,1078,
                1082,1085,1089,1093,1097,1101,1105,1109,1113,1117,1121,1125,1128,1132,1136,1140,1144,1148,1152,1156,1160,
                1164,1167,1171,1175,1179,1183,1187,1191,1195,1199,1203,1207,1210,1214,1218,1222,1226,1230,1234,1238,1242,
                1246,1250,1253,1257,1261,1265,1269,1273,1277,1281,1285,1289,1292,1296,1300,1304,1308,1312,1316,1320,1324,
                1328,1332,1335,1339,1343,1347,1351,1355,1359,1363,1367,1371,1375,1378,1382,1386,1390,1394,1398,1402,1406,
                1410,1414,1417,1421,1425,1429,1433,1437,1441,1445,1449,1453,1457,1460,1464,1468,1472,1476,1480,1484,1488,
                1492,1496,1500,1503,1507,1511,1515,1519,1523,1527,1531,1535,1539,1542,1546,1550,1554,1558};
    }


}
