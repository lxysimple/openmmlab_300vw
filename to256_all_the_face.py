"""
为了还原老师的blur-300vw数据集的高清图,
把老师的图中关键点最左边的点x做crop的x_left,同理,可得y_low, x_right, y_high
然后reshape到256
把原图的最左边的点x做crop的x_left,同理,可得y_low, x_right, y_high
然后reshape到256
这样就能保证两者对齐了,从而能够拿这个样本对训练ESTRNN
"""

import os
from os.path import join
from PIL import Image


# 300vw一共有这么多视频，每个视频都用一个文件夹装着
videos_all =  ['001', '002', '003', '004', '007', '009', '010', '011', '013', '015', 
                    '016', '017', '018', '019', '020', '022', '025', '027', '028', '029', 
                    '031', '033', '034', '035', '037', '039', '041', '043', '044', '046', 
                    '047', '048', '049', '053', '057', '059', '112', '113', '114', '115', 
                    '119', '120', '123', '124', '125', '126', '138', '143', '144', '150', 
                    '158', '160', '203', '204', '205', '208', '211', '212', '213', '214', 
                    '218', '223', '224', '225', 
                                                '401', '402', '403', '404', '405', '406', 
                    '407', '408', '409', '410', '411', '412', '505', '506', '507', '508', 
                    '509', '510', '511', '514', '515', '516', '517', '518', '519', '520', 
                    '521', '522', '524', '525', '526', '528', '529', '530', '531', '533', 
                    '537', '538', '540', '541', '546', '547', '548', '550', '551', '553', 
                    '557', '558', '559', '562']

# Category 1 in laboratory and naturalistic well-lit conditions
videos_test_1 = ['114', '124', '125', '126', '150', '158', '401', '402', '505', '506',
                        '507', '508', '509', '510', '511', '514', '515', '518', '519', '520', 
                        '521', '522', '524', '525', '537', '538', '540', '541', '546', '547', 
                        '548']
# Category 2 in real-world human-computer interaction applications
videos_test_2 = ['203', '208', '211', '212', '213', '214', '218', '224', '403', '404', 
                        '405', '406', '407', '408', '409', '412', '550', '551', '553']

# Category 3 in arbitrary conditions
videos_test_3 = ['410', '411', '516', '517', '526', '528', '529', '530', '531', '533', 
                        '557', '558', '559', '562']

videos_train = [ i for i in videos_all if i not in videos_test_1 
                                        and i not in videos_test_2 
                                        and i not in videos_test_3]

# The broken frames in test dataset
broken_frames = {
    '401':[722,723,724,727,728,729,730,731,788,789,792,795,810,811,850,851,852,853,854,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,897,898,902,903,910,911,913,914,915,916,917,918,919,920,921,922,923,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1140,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1160,1161,1162,1163,1164,1381,1543,1544,1545,1605,1606,1614,1615,1618], 
    '402':[677,678,682,686,687,688,689,690,691,692,693,694,695,805,806,807,808,809,810,811,812,813,814,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1068,1075,1076,1077,1078,1080,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1664,1666,1667,1668,1669,1670,1671,1672,1673,1674,1675,1676,1677,1678,1679,1680,1681,1682,1683,1684,1685,1686,1687,1688,1689,1690,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1701],
    '410':[148,150,176,178,179,180,182,183,184,186,187,188,190,191,192,194,195,196,198,199,200,202,203,204,206,207,208,283,284,286,287,288,290,291,292,294,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,542,543,626,686,687,688,690,691,692,694,695,700,816,823,824,825,826,827,828,829,830,831,832,834,835,1058,1059,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1194,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1232,1234,1235,1239,1240,1242,1243,1282,1283,1284,1286,1287,1288,1290,1291,1292,1294], 
    '411':[19,20,60,80,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,231,232,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,480,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,578,627,628,629,630,631,702,703,707,1271,1272,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289],
    '508':[14,147,148,149,150,267,268,269,270,271,272,273,274,367,368,370,371,372,373,382,613,618,831,832,870,930,931,932,933,934,935,1029,1086,1087,1088,1175,1176,1179,1180,1181,1182,1183,1184,1186,1187,1188,1189,1190,1191,1192,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1316,1501,1502,1927,1928,1930,1931,1932,1933,1934,1935,1973,1974,1987,1989,2008,2036,2039,2040,2042,2043,2044,2045,2046,2239,2382,2404,2405,3018,3123,3124,3125,3126,3127,3128],
    '518':[392,475,476,477,478,479,480,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,514,515,516,517,518,633,670,671,780,781,782,783,784,786,1607,1608,1609,1610,1611,1612,1613,1614,1615,1616,1617,1618,1619,1620,1621,1622,1623,1624,1626,1627,1628,1629,1630,1631,1632,1634,1635,1636,1637,1638,1639,1640,1706,1707,1708,1709,1710,1711,1712,1713,1714,1715,1716,1717,1718,1719,1720,1721,1722,1723,1724,1725,1726,1727,1728,1729,1730,1731,1732,1733,1734,1735,1736,1737,1738,1739,1740,1741,1742,1743,1744,1746,1747,1748,1749,1750,1751,1752,1754,1755,1756,1757,1758,1759,1760,1762,1763,1764,1765,1766,1767,1954,1955,1956,1957,1958,1959,1960,1961,1962,1963,1964,1965,1966,1967,1968,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,2003,2004,2005,2006,2007,2008,2028,2046,2055,2056,2057,2058,2059,2060,2061,2062,2063,2064,2065,2066,2067,2068], 
    '529':[90,91,92,93,94,95,96,133,139,605,644,645,646,647,648,650,651,652,653,654,655,656,659,661,691,692,1340,1341,1342,1343,1344,1346,1347,1348,1349,1357,1367,1368,1370,1371,1372,1373,1374,1375,1376,1422,1423,1424,1483,1824,1908,1909,1910,1911,1912,1914,1915,1916,1917,1918,1919,1920,1922,2058,2059,2060,2061,2062,2063,2075,2076,2077,2078,2079,2080,2226,2227,2228,2248,2250,2251,2252,2253,2254,2255,2256,2258,2259,2260,2261,2262,2263,2264,2266,2267,2268,2288,2290,2422,2423,2424,2426,2427,2428,2429,2430,2431,2432,2434,2435,2436,2437,2438,2439,2440,2442,2443,2444,2484,2487,2498,2499,2500,2501,2502,2503,2834,2835,2836,2837,2838,2839,2840,3007],
    '530':[554,555,556,720,726,727,728,730,731,732,733,734,735,736,738,739,740,741,742,743,744,828,829,830,831,832,834,835,836,837,838,839,840,842,843,844,845,846,847,848,874,875,876,877,878,879,880,882,883,884,885,886,887,888,890,891,892,893,894,895,896,898,899,900,901,927,928,935,936,1186,1187,1188,1247,1248,1250,1276,1296,1356,1357,1358,1359,1360,1361,1426,1427,1802,1803,1820,2074,2075,2076,2077,2078,2079,2080,2082,2083,2084,2085,2086,2087,2088,2090,2091,2092,2093,2094,2095,2096], 
    '531':[109,279,280,281,282,283,284,285,286,287,322,323,324,325],
    '533':[1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1232,1233,1234,1235,1236,1237,1238,1239,1240,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1379,1380,1403,1404,1405,1406,1407,1424,1426,1427,1484,1485,1498,1499,1500,1504,1505],
    '540':[744,746,747,748,749,750], 
    '547':[111,112,114,115,116,117,118,119,120,122,123,124,125,126,127,128,130,131,132,133,134,135,959,978,1050,1051,1052,1053,1054,1055],
    '548':[429,430,431,432,434,435,436,437,438,439,440,442,443,444,445,446,447,448,450,582,583,584,586,587,588,589,590,591,592,594,595,596,597,598,599,600,602,624,626,642,804,805,806,807,808,898,899,900,901,902,903,904,906,907,908,909,910,911,912,914,915,916,917,918,919,920,922,923,924,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200,1201,1202,1203,1204,1205,1206,1207,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1394,1488,1490], 
    '551':[180,181,298,299,300,301,302,494,506,507,508,509,516,562,563,564,565,566,567,568,675,676,677,678,679,680,682,683,684,686,687,688,690,790,791,792],
    '553':[551,552,553,554,555,556,557,558,559,560,713,761,762,763,775,776,778,779,780,783,784,786,787,788,789,790,791,792,808,809,1366],
    '557':[22,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,57,75,177,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,524,536,538,539,544,546,621,622,623,778,779,780,785,793,801,809,820,821,822,994,995,1009,1017,1036,1096,1097,1105,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1305,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1532,1533,1534,1535,1536,1537,1538,1539,1540,1541,1542,1543,1544,1545,1546,1547,1548,1549,1550,1551,1552,1553,1554,1555,1556,1557,1558,1559,1560,1561,1562,1563,1587,1593,1601,1604,1609,1614,1617,1622,1623,1625,1626,1627,1628,1629,1630,1631,1632,1633,1634,1635,1636,1637,1638,1639,1640,1641,1642,1643,1644,1645,1646,1647,1648,1649,1650,1651,1653,1657,1671,1672,1673,1674,1675,1676,1677,1678,1679,1680,1681,1682,1683,1684,1685,1686,1687,1688,1689,1690,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1701,1702,1703,1704,1705,1706,1707,1708,1709,1710,1711,1712,1713,1714,1715]
}

def _keypoint_from_pts_(file_path):
    """
    从一个.pts格式文件中提取68个关键点到列表中,并返回该列表
    return:
        [x1,y1, x2,y2, ..., x68,y68]
    """
    # 创建一个列表来存储关键点坐标
    keypoints = []

    with open(file_path, 'r') as file:
        file_content = file.read()

    # 查找花括号内的数据
    start_index = file_content.find('{')  # 找到第一个左花括号的位置
    end_index = file_content.rfind('}')  # 找到最后一个右花括号的位置

    if start_index != -1 and end_index != -1:
        data_inside_braces = file_content[start_index + 1:end_index]  # 提取花括号内的数据

        # 将数据拆分成行
        lines = data_inside_braces.split('\n')
        for line in lines:
            if line.strip():  # 跳过空行
                x, y = map(float, line.split())  # 假设坐标是空格分隔的
                keypoints.append(x)
                keypoints.append(y)
    else:
        print("未找到花括号内的数据")

    return keypoints


def videos_xy_from_txt(voides, txt_path):
    """
    提取一个.txt，114个视频，所有的帧

    return:
        {
            '001':{
                '0.jpg': [x_left, y_low, x_right, y_high],
                '1.jpg': [x_left, y_low, x_right, y_high],
                ...
            },
            '002':{
                '0.jpg': [x_left, y_low, x_right, y_high],
                '1.jpg': [x_left, y_low, x_right, y_high],
                ...
            },
            ...
        }
    """


    from collections import defaultdict
    outputs = defaultdict(dict) # 定义二维字典

    with open(txt_path, 'r') as file:
        for line in file:

            # 使用 split() 方法将当前行按空格分割成字符串列表
            strings = line.split()
    
            # 检查是否有 69 个字符串
            if len(strings) != 68*2+1:
                print("Error: 每一行应该有 69 个字符串")
                continue
            
            keypoints = []
            path = ''

            keypoints = strings[0:68*2]
            keypoints = list(map(float, keypoints))
            path = strings[68*2]

            
            if path[0:3] not in voides:
                continue
            
            pic_id = int(path[4:-4])

            # if this frame is broken, skip it.
            if path[0:3] in broken_frames and pic_id in broken_frames[path[0:3]]:
                continue
            
            # print('path[0:3] in test.json: ', path[0:3])
            
            keypoints_x = [] 
            keypoints_y = []
            for j in range(68*2):
                if j%2 == 0:
                    keypoints_x.append(keypoints[j])
                else:
                    keypoints_y.append(keypoints[j])
            x_left = min(keypoints_x)  
            x_right = max(keypoints_x) 
            y_low = min(keypoints_y) 
            y_high = max(keypoints_y) 


            outputs[path[0:3]][str(pic_id)] = [x_left, y_low, x_right, y_high] 

            # # 为了和FSTRNN生成结果一致
            # # 丢弃每个序列最后2帧
            # if int(path[4:-4]) == 0 and len(outputs)!=0:
            #     outputs = outputs[:-2]
            # # 丢弃每个序列前2帧    
            # if int(path[4:-4]) == 2 and len(outputs)!=0:
            #     outputs = outputs[:-2]

    # 去掉最后一个序列的最后2帧
    # return outputs[:-2]
    return outputs


def findxy(avideo_annot_path):
    """
    找到某个帧人脸框的左上角坐标
    args:
        avideo_annot_path: 某个帧注解的路径, .../000001.pts
    """
    keypoints = _keypoint_from_pts_(avideo_annot_path)

    keypoints_x = [] 
    keypoints_y = []
    for j in range(68*2):
        if j%2 == 0:
            keypoints_x.append(keypoints[j])
        else:
            keypoints_y.append(keypoints[j])
    x_left = min(keypoints_x)  
    x_right = max(keypoints_x) 
    y_low = min(keypoints_y) 
    y_high = max(keypoints_y) 

    return x_left, y_low, x_right, y_high


def crop_image(png, apic_path, res_path, x_left, y_low, x_right, y_high):
    """
    args:
        png: 图片名
        apic_path: 一个要被裁剪的图片路径, 要求图片命名形式为{:08d}.png
        res_path: 裁剪后结果图所放置的路径
        x_left, y_low, x_right, y_high: 略
    """
    image = Image.open(apic_path)
    cropped_image = image.crop(
                        (x_left, y_low, x_right, y_high)   
                    )
    # 创建注解文件的目录（没有该目录，无法创建注解文件）
    if not os.path.exists(res_path):
        os.makedirs(res_path)


    save_path = join(res_path, png)
    cropped_image.save(save_path)

    return 

def resize256(move, png, apic_path, pic_res_dir):
    """
    args:
        png: 图片名
        apic_path: 要resize的某个帧地址, 要求图片命名形式为{:08d}.png
        pic_res_dir: 输出的图片存放目录
    """

    # 修改图片
    image = Image.open(apic_path)
    image = image.resize((256, 256))

    if not os.path.exists(pic_res_dir):
        os.makedirs(pic_res_dir)

    save_pic = join(pic_res_dir, f"{int(png[:-4])+move:08d}.png") 
    image.save(save_pic)

    return  

def test_300vw():
    # videos = ['001']
    videos = videos_train

    pic_300vw_dir = '/home/xyli/data/300vw'
    annot_300vw_dir = '/home/xyli/data/300VW_Dataset_2015_12_14'

    data300vw_crop_dir_res = '/home/xyli/data/300vw_crop'
    data300vw_resize256_dir_res = '/home/xyli/data/300vw_resize256' 


    for video in videos: # 遍历 [001,002,...]
        # 待转化数据路径
        pngs_dir = join(pic_300vw_dir, video, 'images')
        # pngs_dir = join(pic_300vw_dir, video)
        annots_dir = join(annot_300vw_dir, video, 'annot')
        
        # 转化结果路径
        crop_pic = join(data300vw_crop_dir_res, video)
        # crop_annot = join(data300vw_dir_res, video, 'crop_annot')

        resize_pic = join(data300vw_resize256_dir_res, video)
        # resize_annot = join(data300vw_dir_res, video, 'resize_annot')

        # 如果转化结果路径不存在, 则创建
        if not os.path.exists(crop_pic):
            os.makedirs(crop_pic)
        if not os.path.exists(resize_pic):
            os.makedirs(resize_pic)

        pngs = os.listdir(pngs_dir) 
        
        # from IPython import embed
        # embed()
        for png in pngs: # 遍历 001中的[00000001.png, ...]
            # 某个帧 某个帧注解 路径
            png_path = join(pngs_dir, png)
            annot_path = join(annots_dir, png[-10:-4]+'.pts')

            # 从注解中提取信息
            x_left, y_low, x_right, y_high = findxy(annot_path)
 
            crop_image( 
                png,
                png_path, 
                crop_pic, 
                x_left, y_low, x_right, y_high
            )
            
            # 对crop_image进行二次加工
            png_path = join(crop_pic, png)

            resize256( 
                0,
                png,
                png_path,
                resize_pic,
            )  

        print(f'{video}转化结束！')
        os.system(f'rm -rf {data300vw_crop_dir_res}')

def test_300vw_blur():
    # videos = ['001', '002', '003', '004', '007']
    videos = videos_test_3

    pic_300vw_dir = '/home/xyli/data/Blurred-300VW'
    annot_300vw_dir = '/home/xyli/data/annotations/300VW_blur_label_list_256_valid.txt'

    data300vw_crop_dir_res = '/home/xyli/data/300vw_crop_blur_valid'
    data300vw_resize256_dir_res = '/home/xyli/data/300vw_resize256_blur_valid' 

    videos_xy = videos_xy_from_txt(videos, annot_300vw_dir)

    for video in videos: # 遍历 [001,002,...]
        # 待转化数据路径
        pngs_dir = join(pic_300vw_dir, video,)
        
        # 转化结果路径
        crop_pic = join(data300vw_crop_dir_res, video)
        resize_pic = join(data300vw_resize256_dir_res, video)


        # 如果转化结果路径不存在, 则创建
        if not os.path.exists(crop_pic):
            os.makedirs(crop_pic)
        if not os.path.exists(resize_pic):
            os.makedirs(resize_pic)

        pngs = os.listdir(pngs_dir) 
        
        # from IPython import embed
        # embed()
        for png in pngs: 
            # 某个帧 某个帧注解 路径
            png_path = join(pngs_dir, png)
            # 从注解中提取信息
            x_left, y_low, x_right, y_high = videos_xy[video][png[:-4]]


            # from IPython import embed
            # embed()
            crop_image( 
                png,
                png_path, 
                crop_pic, 
                x_left, y_low, x_right, y_high
            )
            
            # 对crop_image进行二次加工
            png_path = join(crop_pic, png)

            resize256( 
                2,
                png,
                png_path,
                resize_pic,
            )  

        print(f'{video}转化结束！')
        os.system(f'rm -rf {data300vw_crop_dir_res}')

if __name__ == '__main__':
    # test_300vw()
    
    test_300vw_blur()


    

