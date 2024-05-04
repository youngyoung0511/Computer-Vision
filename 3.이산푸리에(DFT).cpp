#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("2.png", IMREAD_GRAYSCALE);

    if (image.empty()) {
        cerr << "Error: Could not read the image." << endl;
        return -1;
    }

    //이미지 행(높이)에 대해 DFT에 최적화된 크기 계산해서 optRows에 저장
    int optRows = getOptimalDFTSize(image.rows);
    //이미지 열(너비)에 대해 DFT에 최적화된 크기 계산해서 optCols에 저장
    int optCols = getOptimalDFTSize(image.cols);

    //getOptimalDFTSize함수는 주어진 크기에 대해 DFT에 가장 효과적인 크기를 계산

    Mat resizedImg; //새로운 이미지를 저장할 resizedImg Mat 객체 선언
    //copyMakeBorder함수를 사용하여 이미지 주위에 0으로 채워진 테두리를 추가하여 이미지 확장
    //BORDER_CONSTANT는 테두리를 상수값으로 채우는 옵션. 테두리 부분이 원본이미지와 연결되지 않고 지정된 상수값으로 채워짐
    //Scalar::all(0)은 0으로 채워진 Scalar 객체
    copyMakeBorder(image, resizedImg, 0, optRows - image.rows, 0, optCols - image.cols, BORDER_CONSTANT, Scalar::all(0));
    //image: 입력이미지, resizedImg: 출력이미지
    //0, optRows - image.rows, 0, optCols - image.cols: 각각 위,아래,왼,오 방향으로 추가할 픽셀 수. 따라서 테두리의 두께 결정


    //channel은 vector<Mat>타입. resizedImg를 푸리에 변환할 때 필요한 두개의 채널을 포함하는 벡터
    //첫번째 채널: resizedImg 그자체(원본이미지)
    //두번째 채널: 크기가 resizedImg와 같고 모든 요소가 0인 행렬(Mat::zeros)
    //두개의 채널이 필요한 이유: 푸리에 변환을 위해선 실수부분, 허수부분이 필요하기 때문
    vector<Mat> channels = { Mat_<float>(resizedImg),Mat::zeros(resizedImg.size(),CV_32F) };
    
    
    Mat complexImg;
    //merge 함수는 여러개의 채널을 단일 다채널 행렬로 병합. 
    //channels벡터에 담딘 두 채널을 하나의 복소수 형태의 이미지인 complexImg로 합침.
    //따라서 complexImg는 복소수 행렬로서 각 픽셀은 실수와 허수 부분을 갖음
    merge(channels, complexImg);
    //dft함수는 푸리에 변환을 수행
    //complexImg를 입력으로 받아서 complexImg에 푸리에 변환 결과 저장

    dft(complexImg, complexImg);


    //복소수 형태의 이미지를 채널로 분리
    //split함수를 사용해서 푸리에 변환된 이미지를 채널로 분리함
    //channels벡터는 실수부분과 허수부부을 나타내는 두개의 채널을 포함
    split(complexImg, channels);

    Mat mag;
    //magnitude함수를 사용하여 주파수 도메인에서의 크기 계산
    //channels[0]과 channels[1]로부터 각 픽셀 위치에서의 크기를 계산하여 mag 행렬에 저장
    //이렇게 하면 mag에는 각 픽셀 위치에서의 주파수 도메인의 크기가 저장됨
    magnitude(channels[0], channels[1],mag);

    //mag 행렬의 모든 픽셀에 1을 더함.
    //1을 더하는 작업을 통해 0인 픽셀에 대한 로그 연산을 수행할 때 무한으로 발산하는 것을 방지.
    //로그함수는 0에 대해 정의되지 않으므로 0에 가까운 값에 1을 더해서 로그 연산이 안전하게 수행
    mag += Scalar::all(1);
    //mag 행렬의 각 픽셀에 대해 자연 로그를 취하고 그 결과를 다시 mag에 저장. 
    //주파수 도메인에서의 크기를 로그 스케일로 변환.
    log(mag, mag);

    //mag 행렬을 0.0에서 1.0 사이의 값으로 정규화
    normalize(mag, mag, 0.0, 1.0, cv::NORM_MINMAX);
    //mag:정규화할 대상 행렬
    //mag:정규화된 결과 저장 행렬
    //0.0:정규화의 최솟값
    //1.0:정규화의 최댓값
    //CV_MINMAX:최소-최대 정규화를 의미함

    //푸리에 변환 결과의 네 사분면을 서로 교환하는 작업
    //푸리에 변환의 결과로 얻은 주파수 도메인에서의 이미지는 네 사분면에 대칭된 형태임
    
    //windowing 함수 적용
    //Mat hannWindow = Mat::zeros(optRows, optCols, CV_32F);
    //createHanningWindow(hannWindow, Size(optCols, optRows), CV_32F);
    //multiply(mag, hannWindow, mag);


    //이미지의 중심 좌표를 계산함.
    //cx:이미지의 가로 중심
    //cy:이미지의 세로 중심
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    //rect 클래스를 이용하여 이미지에서 각 사분면에 해당하는 ROI를 정의
    //Q1:좌상단, Q2:우상단, Q3:좌하단, Q4:우하단에 위치한 네개의 부분영상(ROI)
    Mat Q1(mag, Rect(0, 0, cx, cy));
    Mat Q2(mag, Rect(cx, 0, cx, cy));
    Mat Q3(mag, Rect(0, cy, cx, cy));
    Mat Q4(mag, Rect(cx, cy, cx, cy));

    //각 사분면을 서로 교환하는 작업
    //copyTo함수를 사용하여 임시 행렬 tmp를 생성하고 각 사분면을 서로 교환
    Mat tmp;
    Q1.copyTo(tmp);
    Q4.copyTo(Q1);
    tmp.copyTo(Q4);
    Q2.copyTo(tmp);
    Q3.copyTo(Q2);
    tmp.copyTo(Q3);
    //이렇게 함으로써 푸리에 변환 결과의 네 사분면이 서로 교환되어
    //중심에 주파수가 낮은 부분이 오게 됨
    //푸리에 변환된 결과를 해석하기 쉽도록 이미지의 중심에 주파수 낮은 부분을 위치시킨 것

    if (!image.empty() && !mag.empty())
    {
        imshow("image", image);
        imshow("output", mag);
        waitKey(0);
    }
    else
    {
        cout << "Empty images detected!" << endl;
    }


    return 0;
}
