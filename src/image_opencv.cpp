#ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"
#include<iostream>
#include "typeinfo"
using namespace std;
using namespace cv;

extern "C" {


IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

image mat_to_image(Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{

    // c 는 webcam index
    // f 는 filename pointer  이거 일땐 c,w,h,fps 모두 0
    // w는 
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f); // 동영상파일 일 경우
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w); //webcam 일때
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w); //webcam 일때
    if(fps) cap->set(CV_CAP_PROP_FPS, w); //webcam 일때
    return (void *) cap;
}

image get_image_from_stream(void *p)
{

    //printf("start get_image_stream\n");
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    Mat subImage;
    *cap >> m;
    // 자른 이미지 저장하려는 Mat 변수
    *cap >> subImage;

    if(m.empty()) return make_empty_image(0,0,0);
    //printf("If loop pass\n");


    /* 승훈이형 러닝 돌리기 위해 비디오 프레임 별로 자르기 위해 잠시 사용
    char buf[250] = {0,};
    sprintf(buf, "/home/whatacg/run_data/data%d.jpg", image_num);
    imwrite(buf, m);
    image_num++;
    */

    //cout<<endl;
    /*코드 수정 1차*/
    /*코드 수정 2차*/
    /*코드 수정 3차*/    
//***********************   수정 시작부분 영상 크기 자르기    ****************************//
    int _width_s = 0.25*m.Mat::cols;
    int _width_e = 0.75*m.Mat::cols;
    int _height_s = 0.5*m.Mat::rows;
    int _height_e = m.Mat::rows;
/*  
    cout<<"width_s : "<<_width_s<<endl;
    cout<<"width_e : "<<_width_e<<endl;
    cout<<"height_s : "<<_height_s<<endl;
    cout<<"height_e : " << _height_e<<endl;
*/

    //subImage = m(Range(0.25*_width, 0.75*_width), Range(0.5*_height, _height));
    Rect rect(_width_s,_height_s, _width_e - _width_s , _height_e - _height_s);
    //subImage = m(Range(_width_s, _width_e), Range(_height_s, _height_e));
    subImage = m(rect);
//    cout<<subImage.Mat::total()<<endl;

    return mat_to_image(subImage);
    //********************************************   수정 끝부분     *******************************//

    // 아래가 원본
    //return mat_to_image(m);
}



/*
    // 관심영역 설정

    //im.w im.h
    //X,Y,W,H
    Rect rect(((0.25)*im.w, (0.333)*im.h, (0.5)*im.w, (0.667)*im.h);
    // 관심영역 자르기 (Crop ROI).

    Mat subImage = im(rect);

    image out = load_image_cv(subImage, 3);

*/

/*
//잘못 수정한부분 1
image load_image_cv_custom(char *filename, int channels)
{
    cout << filename <<endl;
    //while(1){}
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;
    Mat subImage;

    m = imread(filename, flag);
    printf("OpenCV file open OK!\n");

    int _width = m.size().width;
    int _height = m.size().height;

    printf("width : %d \n height : %d\n", _width, _height);
    printf("total = %d\n", m.total());
    //subImage = m(Range(100, 300), Range(100, 300));

    printf("W,H OK!\n");

    subImage = m(Range(0.25*_width, 0.75*_width), Range(0.5*_height, _height));

    printf("Cutting OK!\n");


    if(!subImage.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }

    printf("IF loop pass OK!\n");

    image im = mat_to_image(subImage);

    printf("Load image CV OK!");
    return im;

}
*/

image load_image_cv(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;


    m = imread(filename, flag);

    if(!m.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image im = mat_to_image(m);
    return im;

}

int show_image_cv(image im, const char* name, int ms)
{

    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}

void make_window(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL); 
    if (fullscreen) {
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}

// snapshot 찍는 부분
void snap_shot(image im, int left, int top, int right, int bot, float width){
    bool flag = 0;

    Mat snap = image_to_mat(im);
    Rect roi_rect(left, top, (right - left) , (bot-top));
    if (flag == 0){
        imwrite("/home/whatacg/darknet/detect/detected.jpg", snap);
    }
    
    Mat roi_snap = snap(roi_rect);
    if (flag == 0){
        imwrite("/home/whatacg/darknet/detect/snap_detected.jpg", roi_snap);
    }
    flag = 1;
}

/*    
참고
    int _width_s = 0.25*m.Mat::cols;
    int _width_e = 0.75*m.Mat::cols;
    int _height_s = 0.5*m.Mat::rows;
    int _height_e = m.Mat::rows;
    Rect rect(_width_s,_height_s, _width_e - _width_s , _height_e - _height_s);
    return mat_to_image(subImage);*/
/*
void draw_center(image im, int x, int y ){
    Mat m = image_to_mat(im);
    circle(m, Point(x,y), 10, Scalar(0,0,255), -1);
    



}
*/

}

#endif