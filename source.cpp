// OpenCVLoad.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include <cstdio>
#include "opencv.hpp" //opencv 的头文件
#include <opencv2/imgproc.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv; //opencv 的命名空间

//多张图片合并输出函数，dst为输出图像，images为输入图像
void mergeImage(Mat& dst, vector<Mat>& images);
//孔洞填充函数
void fillHole(const Mat srcBw, Mat& dstBw);
//去除小面积区域并输出轮廓
void imbwareaopen(const Mat srcBw, Mat& dstBw, Mat& dstboundary, int size);

int main()
{
    printf("Hello Open CV!");
	vector<Mat> images(4);
	Mat dst;

    Mat r = imread("C:/Users/1/Desktop/毕业设计/毕业设计/1.bmp");  //读取图片
	Mat r_gray;
	cvtColor(r, r_gray, COLOR_BGR2GRAY);//转化为灰度图像

	images[0] = r_gray.clone();

	/*自适应二值化*/
	Mat x;
	adaptiveThreshold(r_gray,x,255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,99,1);//均值二值
	//resize(x, x, Size(400, 300));
	//imshow("1", x);

	Mat k;
	adaptiveThreshold(r_gray, k, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 99, 1);//高斯二值
	//resize(k, k, Size(400, 300));
	//imshow("2", k);

	/*开闭*/
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat m = x; 
	int n = 1;//开闭操作次数
	for (int i = 0; i < n; i++)
	{
		morphologyEx(m, m, MORPH_OPEN, kernel);//开操作
		morphologyEx(m, m, MORPH_CLOSE, kernel);//闭操作
	}
	//imshow("开闭", m);


	/*膨胀*/
	n = 1;//膨胀操作次数
	for (int i = 0; i < n; i++)
	{
		dilate(m,m,kernel);//膨胀操作
	}
	//imshow("膨胀", m);


	/*孔洞填充*/
	fillHole(m, m);//孔洞填充操作
	//imshow("孔洞填充", m);
	images[1] = m.clone();


	/*删除小面积对象并输出轮廓	*/
	Mat boundary = Mat::zeros(m.size(), CV_8UC1);//
	imbwareaopen(m, m, boundary, 6000);//删除小面积对象并输出轮廓
	//imshow("删除小面积", m);
	//imshow("轮廓", boundary);
	images[2] = m.clone();
	images[3] = boundary.clone();


	//resize(m, m, Size(400, 300));
	//imshow("3", m);

	/*合并输出展示*/
	mergeImage(dst, images);
	imshow("dst", dst);

    waitKey(0);

    return 0;
}

void mergeImage(Mat& dst, vector<Mat>& images)
{
	int imgCount = (int)images.size();

	if (imgCount <= 0)
	{
		printf("the number of images is too small\n");
		return;
	}

	printf("imgCount = %d\n", imgCount);

	/*将每个图片缩小为指定大小*/
	int rows = 450;
	int cols = 600;
	for (int i = 0; i < imgCount; i++)
	{
		resize(images[i], images[i], Size(cols, rows)); //注意区别：Size函数的两个参数分别为：宽和高，宽对应cols，高对应rows
	}

	/*创建新图片的尺寸
		高：rows * imgCount/2
		宽：cols * 2
	*/
	dst.create(rows * imgCount / 2, cols * 2, CV_8UC3);
	cvtColor(dst, dst, COLOR_BGR2GRAY);

	for (int i = 0; i < imgCount; i++)
	{
		images[i].copyTo(dst(Rect((i % 2) * cols, (i / 2) * rows, cols, rows)));
	}
}

void fillHole(const Mat srcBw, Mat& dstBw)
{
	Size m_Size = srcBw.size();
	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());//延展图像
	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

	cv::floodFill(Temp, Point(0, 0), Scalar(255));

	Mat cutImg;//裁剪延展的图像
	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

	dstBw = srcBw | (~cutImg);
}



void imbwareaopen(const Mat srcBw, Mat& dstBw, Mat& dstboundary, int size)
{
	vector<vector<Point>> contours;           //二值图像轮廓的容器
	vector<Vec4i> hierarchy;                  //4个int向量，分别表示后、前、父、子的索引编号

	findContours(srcBw, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);             //检测所有轮廓

	Mat ImageContours = Mat::zeros(srcBw.size(), CV_8UC1);  //绘制
	Mat ImgContours = Mat::zeros(srcBw.size(), CV_8UC1);


	vector<vector<Point>>::iterator k;                    //迭代器，访问容器数据

	for (k = contours.begin(); k != contours.end();)      //遍历容器,设置面积因子
	{
		if (contourArea(*k, false) < size)
		{//删除指定元素，返回指向删除元素下一个元素位置的迭代器
			k = contours.erase(k);
		}
		else
			++k;
	}

	//contours[i]代表第i个轮廓，contours[i].size()代表第i个轮廓上所有的像素点
	for (int i = 0; i < contours.size(); i++)
	{
		for (int j = 0; j < contours[i].size(); j++)
		{
			//获取轮廓上点的坐标
			Point P = Point(contours[i][j].x, contours[i][j].y);
			ImgContours.at<uchar>(P) = 255;
		}
		drawContours(ImageContours, contours, i, Scalar(255), -1, 8);
	}

	dstBw = ImageContours;
	//imshow("删除小面积", ImageContours);
	dstboundary = ImgContours;
	//imshow("轮廓点集合", ImgContours);
	waitKey(0);

}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
