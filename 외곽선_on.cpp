#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    // 카메라 열기
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cerr << "Error: 카메라를 열 수 없습니다." << endl;
        return -1;
    }

    while (true) {
       
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            cerr << "Error: 프레임을 읽을 수 없습니다." << endl;
            break;
        }

        Mat image2 = frame.clone();

        // Image preprocessing
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(5, 5), 0);
        bitwise_not(gray, gray);

        // Canny edge detection
        Mat edge;
        Canny(gray, edge, 100, 100);

        // Thresholding
        Mat thresh;
        threshold(gray, thresh, 127, 255, THRESH_BINARY);

        // Adaptive thresholding
        Mat adaptive_threshold;
        adaptiveThreshold(gray, adaptive_threshold, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

        // Morphological operations
        erode(thresh, thresh, Mat(), Point(-1, -1), 2);
        dilate(thresh, thresh, Mat(), Point(-1, -1), 2);

        // Find contours
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(thresh.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        

        for (size_t i = 0; i < contours.size(); ++i) {
            const auto& contour = contours[i];

            // Draw contours
            drawContours(frame, vector<vector<Point>>{contour}, -1, Scalar(0, 255, 0), 3); 
        }
           

        // Display results
        imshow("contour", frame);

        // 키 입력 확인 (ESC 키를 누르면 종료)
        if (waitKey(1) == 27) {
            break;
        }
    }

    return 0;
}
