#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat image = imread("10.jpg");
    //Mat image2 = image.clone();

    //흑백으로 변환
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    //가우시안 블러-노이즈 제거
    GaussianBlur(gray, gray, Size(5, 5), 0);
   

    //캐니 에지 검출을 사용하여 이미지에서 에지 찾음
    Mat edge;
    Canny(gray, edge, 100, 100);

    //이진화 수행
    Mat thresh;
    threshold(gray, thresh, 127, 255, THRESH_BINARY);

 
    //모폴로지 연산을 사용. 노이즈 제거, 이미지 정리. 객체 구멍 메우기 
    //이미지 침식 erode. 노이즈 제거, 객체 크기 즐이기: 흰색 객체의 주변을 검은색으로 채우면서 객체 줄이는 효과
    erode(thresh, thresh, Mat(), Point(-1, -1), 2);
    //이미지 팽창 dilate. 객체 크기 기우거나 끊어진 부분 연결. 흰색 객체의 주변을 검은색으로 채우면서 객체 확장
    dilate(thresh, thresh, Mat(), Point(-1, -1), 2);

    //이진화된 이미지에서 윤곽선 찾기
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresh.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


    //윤곽선을 이미지에 그림
    for (size_t i = 0; i < contours.size(); i++) {
        const auto& contour = contours[i];
        drawContours(image, vector<vector<Point>>{contour}, -1, Scalar(0, 255, 0), 3);
       
    }

    imshow("contour", image);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
