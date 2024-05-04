#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main()
{
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
        Mat frame;
        cap >> frame; // 카메라로부터 한 프레임을 읽어옴

        if (frame.empty())
        {
            cout << "프레임을 읽을 수 없습니다." << endl;
            break;
        }

        Mat gray, edges, cdst;

        // 그레이스케일 변환
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 가우시안 블러링 적용
        GaussianBlur(gray, gray, Size(9, 9), 2, 2);

        vector<Vec3f> circles;

        // 원을 검출하는 허프 변환
        HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 8, 200, 50, 0, 0);

        // 원을 영상 위에 그린다. 
        for (size_t i = 0; i < circles.size(); i++)
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            circle(frame, center, 3, Scalar(0, 255, 0), -1, 8, 0); // 원의 중심 그리기
            circle(frame, center, radius, Scalar(0, 0, 255), 3, 8, 0); // 원을 그리기
        }

        // 결과 표시
        imshow("Live", frame);

        // 'ESC' 키를 누르면 종료
        if (waitKey(30) == 27)
            break;
    }

    return 0;
}
