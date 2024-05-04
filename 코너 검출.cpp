#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat src, gray;
    int thresh = 150;
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    // 카메라 열기 (0은 기본 카메라, 여러 개일 경우 1, 2, ...)
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "카메라를 열 수 없습니다." << endl;
        return -1;
    }

    namedWindow("Live", 1);

    while (true)
    {
        // 프레임 읽기
        cap >> src;

        // 그레이스케일 변환
        cvtColor(src, gray, COLOR_BGR2GRAY);

        Mat dst, dst_norm, dst_norm_scaled;
        dst = Mat::zeros(src.size(), CV_32FC1);

        // Harris 코너 검출
        cornerHarris(gray, dst, blockSize, apertureSize, k);

        normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
        convertScaleAbs(dst_norm, dst_norm_scaled);

        for (int j = 0; j < dst_norm.rows; j++)
        {
            for (int i = 0; i < dst_norm.cols; i++)
            {
                if ((int)dst_norm.at<float>(j, i) > thresh)
                {
                    circle(src, Point(i, j), 5, Scalar(0, 0, 255), 2, 8, 0);
                }
            }
        }

        // 결과 표시
        imshow("코너 검출", src);

        // 'ESC' 키를 누르면 종료
        if (waitKey(30) == 27)
            break;
    }

    return 0;
}
