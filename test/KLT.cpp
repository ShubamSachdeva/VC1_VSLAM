//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std;


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
int main( int argc, char** argv )
{

    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- Read two images
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    const int grid_width = 40;
    const int grid_height = 30;
    const float resolution = img_1.rows/grid_height;  
    const float dis_threshold = 0.02;
    cout<<"the rows of img"<<img_1.rows<<endl;
    cout<<"the rows of img"<<img_1.cols<<endl;

    list< cv::Point2f > keypoints;
    vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
	detector->set("thres", 100);
    bool grid[grid_width][grid_height]; //initialize all the cells to non-occupied
    for (int i =0; i<grid_height; i++)
        for (int j=0; j<grid_width; j++)
            grid[i][j]=false;

    // for (int i =0; i<grid_height; i++)
    //     for (int j=0; j<grid_width; j++)
    //         cout<<grid[i][j]<<endl;

    detector->detect( img_1, kps );
    for ( auto kp:kps ){
        int x = kp.pt.x/resolution;
        int y = kp.pt.y/resolution;
        cout<<"x "<<x<<" y "<<y<<" resolution "<<resolution<<endl;
        if(grid[x][y] == false)
        {
            keypoints.push_back( kp.pt );
            grid[x][y]= true;
        }
    }
       
    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    vector<cv::Point2f> prev_keypoints_bkwd;
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;
    vector<unsigned char> status1;
    vector<float> error1;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
    cout<<prev_keypoints.size()<<endl;
    cv::calcOpticalFlowPyrLK( img_2, img_1, next_keypoints, prev_keypoints_bkwd, status1, error1 );
    cout<<"""""""""aaaaaaaaaaaaaaaaabbbbbbbbbbb\n"<<prev_keypoints_bkwd.size()<<endl;

    // auto dis = norm(prev_keypoints[0]-prev_keypoints_bkwd[0]);
    // cout<<dis<<endl;

    for (int i=0; i<prev_keypoints.size(); i++)
    {
        auto dis = norm(prev_keypoints[i]-prev_keypoints_bkwd[i]);
        cout<<"distance is "<<dis<<endl;
        if (dis>dis_threshold)
            status[i] = 0;

    }
    
    
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    // visualize all  keypoints
    hconcat(img_1,img_2,img_1);
    for ( size_t i=0; i< prev_keypoints.size() ;i++)
    {
        //cout<<(int)status[i]<<endl;
        if(status[i] == 1)
        {
            Point pt;
            pt.x =  next_keypoints[i].x + img_2.size[1];
            pt.y =  next_keypoints[i].y;

            line(img_1, prev_keypoints[i], pt, cv::Scalar(0,255,255));
        }
    }

    cv::imshow("klt tracker", img_1);
    cv::waitKey(0);

    return 0;
}
