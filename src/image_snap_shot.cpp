//************************** 스냅샷 저장후 넘기는 모듈 ***********************//
#ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "darknet.h"
#include "opencv2/opencv.hpp"
#include "image.h"
#include<iostream>
#include "typeinfo"
using namespace std;
//using namespace Mat;

using namespace cv;
bool flag = 0;

//extern "C" {





    // 차선 침범시 플래그 받는 부분
    /*
    bool get_signal(){
        // 통신 해서 플래그만 받아오는 부분
        // 플래그 받는 순간 시간 체크.....
        // 스냅샷 시그널 넣기
        // 일단 간단하게 만들기
        /* 1차 메이킹 */
/*
        if(flag){
            for(long long i = 0; i<1000000000000000000; i++){
            }

            snap_shot();
        }
    }
    */
    
    // 찍은 사진 보내주기
   // void comm(){

  //  }
    

    // 스냅샷 찍는 부분
    // 리턴값은...... Mat 형태.....
    /*
    void snap_shot(image im, int left, int top, int right, int bot, float width){
        if (flag = 0)
            cv2.imwrite("/home/whatacg/darknet/detect/detected.jpg", im);
        flag = 1;
    }
*/



//}




#endif