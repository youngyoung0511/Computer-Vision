#include<iostream>
#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

int main()
{
	Mat scr, scr_gray;

	scr = imread("8.jpg", 1);
	imshow("scr", scr);

	//그레이 스케일로 변환
	cvtColor(scr, scr_gray, cv::COLOR_BGR2GRAY);

	//가우시안 블러링 적용
	GaussianBlur(scr_gray, scr_gray, Size(9, 9), 2, 2);

	vector<Vec3f> circles;

	//원을 검출하는 허프 변환
	HoughCircles(scr_gray, circles, cv::HOUGH_GRADIENT, 1, scr_gray.rows / 8, 200, 50, 0, 0);

	//원을 영상 위에 그린다. 
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(scr, center, 3, Scalar(0, 255, 0), -1, 8, 0); //원의 중심 그리기
		circle(scr, center, radius, Scalar(0, 0, 255), 3, 8, 0);//원을 그리기
	}

	imshow("원형 허프 변환", scr);
	waitKey(0);
	return 0;
}
