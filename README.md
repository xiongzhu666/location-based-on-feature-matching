# location-based-on-feature-matching

该项目基于GMS特征匹配，并在此基础上进行png地图的机器定位实现，GMS原理参考开源项目：https://github.com/JiawangBian/GMS-Feature-Matcher

Requirement: OpenCV 3.0 or later (for IO and ORB features, necessary)

C++ Example:

./img_location ../data/location_7.png ../data/floor_1.png 200 200 9

./img_location ../data/test.png ../data/ref_map.png 200 500 9

location_7.png: 子图png图像

floor_1.png: 全局png地图

200: 机器在子图中的横坐标（左上角为点{0，0}）

200: 机器在子图中的纵坐标（左上角为点{0，0}）

9: GMS匹配可调参数（值越大，匹配点越少，反之则反）
