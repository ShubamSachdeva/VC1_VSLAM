////
//// Created by sicong on 08/11/18.
////
//
//#include <iostream>
//#include <fstream>
//#include <list>
//#include <vector>
//#include <chrono>
//using namespace std;
//
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/video/tracking.hpp>
//
//using namespace cv;
//int main( int argc, char** argv )
//{
//
//    if ( argc != 3 )
//    {
//        cout<<"usage: feature_extraction img1 img2"<<endl;
//        return 1;
//    }
//    //-- Read two images
//    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
//    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
//
//    list< cv::Point2f > keypoints;
//    vector<cv::KeyPoint> kps;
//
//    std::string detectorType = "Feature2D.BRISK";
//    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
//	detector->set("thres", 100);
//
//
//    detector->detect( img_1, kps );
//    for ( auto kp:kps )
//        keypoints.push_back( kp.pt );
//
//    vector<cv::Point2f> next_keypoints;
//    vector<cv::Point2f> prev_keypoints;
//    for ( auto kp:keypoints )
//        prev_keypoints.push_back(kp);
//    vector<unsigned char> status;
//    vector<float> error;
//    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
//    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
//    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
//
//    // visualize all  keypoints
//    hconcat(img_1,img_2,img_1);
//    for ( int i=0; i< prev_keypoints.size() ;i++)
//    {
//        cout<<(int)status[i]<<endl;
//        if(status[i] == 1)
//        {
//            Point pt;
//            pt.x =  next_keypoints[i].x + img_2.size[1];
//            pt.y =  next_keypoints[i].y;
//
//            line(img_1, prev_keypoints[i], pt, cv::Scalar(0,255,255));
//        }
//    }
//
//    cv::imshow("klt tracker", img_1);
//    cv::waitKey(0);
//
//    return 0;
//}


//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <math.h>
using namespace std;


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;

bool checkinlier(cv::Point2f prev_keypoint,cv::Point2f next_keypoint,cv::Matx33d Fcandidate,double d){
    //fill the blank
    Matx31d X1;
    //cout<<"original"<<next_keypoint;
    X1(0,0)=prev_keypoint.x;
    X1(0,1)=prev_keypoint.y;
    X1(0,2)=1.0;
    //cout<<"after in Matx"<<X1<<endl;
    auto epipolar_line = Fcandidate.t()*X1;
    //cout<<"value of epipolar line"<<epipolar_line<<endl;
    //cout<<"epipolar line "<<epipolar_line<<endl;
    float a = epipolar_line(0,0);
    float b = epipolar_line(1,0);
    float c = epipolar_line(2,0);
    float u = next_keypoint.x;
    float v = next_keypoint.y;
    float dist = abs(a*u+b*v+c)/sqrt(a*a+b*b);
    
    if(dist<=d)
    {
        //cout<<"distance "<<dist<<endl;
        return true;
    }      
    else
        return false;
}

cv::Matx33d Findfundamental(vector<cv::Point2f> prev_subset,vector<cv::Point2f> next_subset){
    Matx33d F;
    Mat A(prev_subset.size(), 9, CV_64FC1);
    Mat X1T(1, 3, CV_64FC1);
    Mat X2(3, 1, CV_64FC1);

    Mat N(3,3, CV_64FC1); //Normalization matrix
    N.at<double>(0,0) = 2.0/640.0;
    N.at<double>(0,1)=0;
    N.at<double>(0,2)=-1;
    N.at<double>(1,0)=0;
    N.at<double>(1,1)=2.0/480.0;
    N.at<double>(1,2)=-1;
    N.at<double>(2,0)=0;
    N.at<double>(2,1)=0;
    N.at<double>(2,2)=1;
    //cout<<"prev_sub "<<prev_subset.size()<<endl;
    //cout<<"next_subset "<<next_subset.size()<<endl;

    for (size_t i=0; i<prev_subset.size(); i++)
    {
        
        X1T.at<double>(0,0)=prev_subset[i].x;
        X1T.at<double>(0,1)=prev_subset[i].y;
        X1T.at<double>(0,2)=1.0;
        X1T = (N * X1T.t()).t();
        //cout<<i<<"th"<<" X1T "<<X1T<<endl;
        X2.at<double>(0,0)=next_subset[i].x;
        X2.at<double>(1,0)=next_subset[i].y;
        X2.at<double>(2,0)=1.0;

        X2 = N*X2;
        //cout<<i<<"th"<<" X2 "<<X2<<endl;

        Mat temp = X1T.t()*X2.t();
        Mat temp2(1, 9, CV_64FC1);
        A.at<double>(i,0)=temp.at<double>(0,0);
        A.at<double>(i,1)=temp.at<double>(0,1);
        A.at<double>(i,2)=temp.at<double>(0,2);
        A.at<double>(i,3)=temp.at<double>(1,0);
        A.at<double>(i,4)=temp.at<double>(1,1);
        A.at<double>(i,5)=temp.at<double>(1,2);
        A.at<double>(i,6)=temp.at<double>(2,0);
        A.at<double>(i,7)=temp.at<double>(2,1);
        A.at<double>(i,8)=temp.at<double>(2,2);

        // A.push_back(temp2);
        // cout<<i<<"th  A "<<A.size()<<"======="<<endl;
    }
    SVD svd(A);
    cv::Mat vt(9, 1, CV_64FC1);

    //cout<<"svd u"<<svd.u<<endl;
    //cout<<"svd w"<<svd.w<<endl;
    //cout<<"svd vt"<<svd.vt<<endl;
  
    cv::Mat f_test(3,3, CV_64FC1);
    //cout<<"f_test size"<<svd.vt.rows<<endl;
    f_test.at<double>(0,0)=svd.vt.at<double>(svd.vt.rows-1,0);
    f_test.at<double>(0,1)=svd.vt.at<double>(svd.vt.rows-1,1);
    f_test.at<double>(0,2)=svd.vt.at<double>(svd.vt.rows-1,2);
    f_test.at<double>(1,0)=svd.vt.at<double>(svd.vt.rows-1,3);
    f_test.at<double>(1,1)=svd.vt.at<double>(svd.vt.rows-1,4);
    f_test.at<double>(1,2)=svd.vt.at<double>(svd.vt.rows-1,5);
    f_test.at<double>(2,0)=svd.vt.at<double>(svd.vt.rows-1,6);
    f_test.at<double>(2,1)=svd.vt.at<double>(svd.vt.rows-1,7);
    f_test.at<double>(2,2)=svd.vt.at<double>(svd.vt.rows-1,8);
    
    //cout<<X2*f_test*X1<<endl;
    //cout<<"f_test matrix"<<f_test<<"============"<<endl;

    SVD svd_F(f_test); //apply rank2 constrain
    //cout<<"test============"<<svd_F.w<<endl;
    Mat sigma_2 = Mat::zeros(3,3, CV_64FC1);
    sigma_2.at<double>(0,0) = svd_F.w.at<double>(0,0);
    sigma_2.at<double>(1,1) = svd_F.w.at<double>(1,0);
    f_test = svd_F.u * sigma_2 * svd_F.vt;
    f_test = (N.t()*f_test*N);
    //cout<<"sigma_2============"<<sigma_2<<endl;
    F= f_test;

    // cout<<"verification!!!!!!!! "<<X1T*f_test*X2<<endl;
    //  cout<<"verification!!!!!!!! "<<F<<endl;
    // cout<<"=============start calculating distance========="<<endl;
    // for(int i=0; i<prev_subset.size(); i++)
    // {   
    //     checkinlier(prev_subset[i], next_subset[i], F, 1.5f);
    // }
    // cout<<"=============end calculating distance========="<<endl;
    return F;
}

void vizEpipolarConstrain(Mat img_1, Mat img_2, vector<cv::Point2f> prev_keypoints,vector<cv::Point2f> next_keypoints, Matx33d f)
{
   // visualize all  keypoints
   hconcat(img_2,img_1,img_1);
   for ( size_t i=0; i< prev_keypoints.size() ;i++)
   {
           Matx31d X1;
            X1(0,0)=prev_keypoints[i].x;
            X1(0,1)=prev_keypoints[i].y;
            X1(0,2)=1.0;
            auto epipolar_line = f.t()*X1;
            double a = epipolar_line(0,0);
            double b = epipolar_line(1,0);
            double c = epipolar_line(2,0);
            Point pt1, pt2;
            pt1.x =img_2.size[1];
            pt1.y =-c/b;
            pt2.x = img_2.size[1]+img_2.size[1];
            pt2.y = -(c+a*img_2.size[1])/b;
            line(img_1, pt1, pt2, cv::Scalar(0,255,255));
            circle(img_1, Point(prev_keypoints[i].x+img_2.size[1], prev_keypoints[i].y), 4, cv::Scalar(0,255,160),CV_FILLED);

            Matx31d X2;
            X2(0,0)=next_keypoints[i].x;
            X2(0,1)=next_keypoints[i].y;
            X2(0,2)=1.0;
            epipolar_line = f*X2;
            a = epipolar_line(0,0);
            b = epipolar_line(1,0);
            c = epipolar_line(2,0);
            Point pt3, pt4;
            pt3.x =0;
            pt3.y =-c/b;
            pt4.x = img_2.size[1];
            pt4.y = -(c+a*img_2.size[1])/b;

           line(img_1, pt3, pt4, cv::Scalar(160,0,255));
           circle(img_1, next_keypoints[i], 4, cv::Scalar(255,0,160),CV_FILLED);
       
   }

   cv::imshow("klt tracker", img_1);
   cv::waitKey(0);

}

void findPose(const Matx33d &E, Matx44d &P1, Matx44d &P2, Matx44d &P3, Matx44d &P4)
{
    Matx33d W(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    Matx33d Z(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    SVD svd(E);
    // auto S1 = -svd.u * Mat(Z) * svd.u.t();
    Mat R1 = svd.u * Mat(W).t() * svd.vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    Mat R2 = svd.u * Mat(W) * svd.vt;
    if(cv::determinant(R2)<0)
        R2=-R2;

    double scal = (svd.w.at<double>(0,0)+svd.w.at<double>(1,0))/2; 
    cout<<"svd(E) w"<<svd.w<<endl;
    cout<<"scalar "<<scal<<endl;
    Mat S = svd.u * Mat(Z) * svd.u.t();
    cout<<"S "<<S<<endl;
    S = S * scal;
    cout<<"S "<<S<<endl;

    SVD svd_S(S);
    cout<<"svd_S.vt "<<svd_S.vt<<endl;
    Mat u3(3, 1, CV_64FC1);
    u3.at<double>(0,0) = svd_S.vt.at<double>(2,0);
    u3.at<double>(1,0) = svd_S.vt.at<double>(2,1);
    u3.at<double>(2,0) = svd_S.vt.at<double>(2,2);
    //P1=====
    for (int i=0; i<3; i++)
    {
        for(int j = 0; j<3; j++)
            P1(i,j) = R2.at<double>(i,j);
        P1(i,3) = u3.at<double>(i,0);
    } 
    P1(3,0)=0.0;
    P1(3,1)=0.0;
    P1(3,2)=0.0;
    P1(3,3)=1.0;

    //P2========
    for (int i=0; i<3; i++)
    {
        for(int j = 0; j<3; j++)
            P2(i,j) = R1.at<double>(i,j);
        P2(i,3) = u3.at<double>(i,0);
    } 
    P2(3,0)=0.0;
    P2(3,1)=0.0;
    P2(3,2)=0.0;
    P2(3,3)=1.0;

    //P3========
    for (int i=0; i<3; i++)
    {
        for(int j = 0; j<3; j++)
            P3(i,j) = R2.at<double>(i,j);
        P3(i,3) = -u3.at<double>(i,0);
    }
    P3(3,0)=0.0;
    P3(3,1)=0.0;
    P3(3,2)=0.0;
    P3(3,3)=1.0;

    //P4========
    for (int i=0; i<3; i++)
    {
        for(int j = 0; j<3; j++)
            P4(i,j) = R1.at<double>(i,j);
        P4(i,3) = -u3.at<double>(i,0);
    }
    P4(3,0)=0.0;
    P4(3,1)=0.0;
    P4(3,2)=0.0;
    P4(3,3)=1.0;
    cout<<"P1 is "<<P1<<endl;
    cout<<"P2 is "<<P2<<endl;
    cout<<"P3 is "<<P3<<endl;
    cout<<"P4 is "<<P4<<endl;
    cout<<"-u3 "<<-u3<<endl;
   
}

int find3DX(Matx44d P, vector<cv::Point2f> prev_point, vector<cv::Point2f> next_point , Matx33d K)
{
    Matx34d Proj(1, 0, 0, 0, 0 ,1, 0, 0, 0, 0, 1, 0);
    Matx44d R(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    Matx44d Translation(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    Matx34d P1 = K*Proj*R*Translation;
    //cout<<"P1 "<<P1<<endl;
    int count=0;
    // cout<<"prev_point.x "<<prev_point.x<<endl;
    // cout<<"original P1.row(2) "<<P1.row(2)<<endl;
    // cout<<"P1.row(2) "<<prev_point.x*P1.row(2)<<endl;
    // cout<<"P1.row(0) "<<P1.row(0)<<endl;

    // cout<<"aaaaaaaaaaa"<<Mat(prev_point.x*P1.row(2) - P1.row(0))<<endl;
    for(size_t i=0; i<prev_point.size(); i++)
    {
        Mat A(4,4,CV_64FC1) ;
        A.row(0) = prev_point[i].x*Mat(P1.row(2)) - Mat(P1.row(0));
        A.row(1) = prev_point[i].y*Mat(P1.row(2)) - Mat(P1.row(1));
        A.row(2) = next_point[i].x*Mat((K*Proj*P).row(2)) - Mat((K*Proj*P).row(0));
        A.row(3) = next_point[i].y*Mat((K*Proj*P).row(2)) - Mat((K*Proj*P).row(1));

        SVD svd(A);

        //cout<<svd.vt<<endl;
        //cout<<"check whether minus "<< svd.vt.at<double>(3,2)/svd.vt.at<double>(3,3)<<endl;
        //cout<<"P1 "<<P1<<endl;

        //cout<<"A "<<A<<endl;
        double a = svd.vt.at<double>(3,0);
        double b = svd.vt.at<double>(3,1);
        double c = svd.vt.at<double>(3,2);
        double d = svd.vt.at<double>(3,3);

        Matx41d x_w(a, b, c, d);
        Matx41d x_c = P * x_w;
        //cout<<"Xc is "<<x_c<<endl;
        if((c/d)>0 && (x_c(0,2)/x_c(0,3))>0)
        {
            count++;
        }

        }
    return count;
}

int main( int argc, char** argv )
{

    srand ( time(NULL) );

    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- Read two images
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    list< cv::Point2f > keypoints;
    vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
    detector->set("thres", 100);

    detector->detect( img_1, kps );
    for ( auto kp:kps )
        keypoints.push_back( kp.pt );

    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    vector<cv::Point2f> kps_prev,kps_next;
    kps_prev.clear();
    kps_next.clear();
    for(size_t i=0;i<prev_keypoints.size();i++)
    {
        if(status[i] == 1)
        {
            kps_prev.push_back(prev_keypoints[i]);
            kps_next.push_back(next_keypoints[i]);
        }
    }


    // p Probability that at least one valid set of inliers is chosen
    // d Tolerated distance from the model for inliers
    // e Assumed outlier percent in data set.
    double p = 0.99;
    double d = 1.5f;
    double e = 0.2;

    int niter = static_cast<int>(std::ceil(std::log(1.0-p)/std::log(1.0-std::pow(1.0-e,8))));
    Mat Fundamental;
    cv::Matx33d F,Fcandidate;
    int bestinliers = -1;
    vector<cv::Point2f> prev_subset,next_subset;
    int matches = kps_prev.size();
    prev_subset.clear();
    next_subset.clear();

for(int i=0;i<niter;i++){
        // step1: randomly sample 8 matches for 8pt algorithm
        unordered_set<int> rand_util;
        while(rand_util.size()<8)
        {
            int randi = rand() % matches;
            rand_util.insert(randi);
        }
        vector<int> random_indices (rand_util.begin(),rand_util.end());
        for(size_t j = 0;j<rand_util.size();j++){
            prev_subset.push_back(kps_prev[random_indices[j]]);
            next_subset.push_back(kps_next[random_indices[j]]);
        }   
        // step2: perform 8pt algorithm, get candidate F
        Fcandidate = Findfundamental(prev_subset,next_subset);
        // step3: Evaluate inliers, decide if we need to update the best solution
        int inliers = 0;
        for(size_t j=0;j<kps_prev.size();j++){
            if(checkinlier(kps_prev[j],kps_next[j],Fcandidate,d))
                inliers++;
        }
        cout<<i<<" th inliers"<<inliers<<endl;
        if(inliers > bestinliers)
        {
            F = Fcandidate;
            bestinliers = inliers;
        }
        prev_subset.clear();
        next_subset.clear();
    }

    // step4: After we finish all the iterations, use the inliers of the best model to compute Fundamental matrix again.
    for(size_t j=0;j<kps_prev.size();j++){
        if(checkinlier(kps_prev[j],kps_next[j],F,d))
        {
            prev_subset.push_back(kps_prev[j]);
            next_subset.push_back(kps_next[j]);
        }

    }
    F = Findfundamental(prev_subset,next_subset);
    

    // cout<<"Fundamental matrix is \n"<<F<<endl;
    // cout<<"Fundamental matrix from cv is \n"<<F2<<endl;
    //vizEpipolarConstrain(img_1, img_2, prev_subset, next_subset, F);
    
    //test embended F================
    Matx33d F2 = findFundamentalMat(kps_prev, kps_next, FM_RANSAC, 1.5f, 0.99);
    // prev_subset.clear();
    // next_subset.clear();
    // for(size_t j=0;j<kps_prev.size();j++){
    //     if(checkinlier(kps_prev[j],kps_next[j],F2,d))
    //     {
    //         prev_subset.push_back(kps_prev[j]);
    //         next_subset.push_back(kps_next[j]);
    //     }

    // }
    // vizEpipolarConstrain(img_1, img_2, kps_prev, kps_next, F2);
    //end testing===============


    FileStorage fs("../config/default.yaml", FileStorage::READ);
    // string aaa = fs["dataset_dir"];
    // fs.release();
    double fx = fs["camera.fx"];
    double fy = fs["camera.fy"];
    double cx = fs["camera.cx"];
    double cy = fs["camera.cy"];
    Matx33d K(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    Matx33d E = K.t() * F2 * K;
    Matx44d P1, P2, P3, P4;
    findPose(E, P1, P2, P3, P4);
    
    cout<<"P1 "<<P1<<endl;
    cout<<"P2 "<<P2<<endl;
    cout<<"P3 "<<P3<<endl;
    cout<<"P4 "<<P4<<endl;


    cout<<find3DX(P1, prev_subset, next_subset, K)<<endl;
    cout<<find3DX(P2, prev_subset, next_subset, K)<<endl;
    cout<<find3DX(P3, prev_subset, next_subset, K)<<endl;
    cout<<find3DX(P4, prev_subset, next_subset, K)<<endl;

    cout<<"Fundamental matrix\n"<<K<<endl;
    cout<<"Essential matrix"<<E<<endl;
    //Matx33d W(0, -1.0, 0, 1.0, 0, 0, 0, 0, 1.0);
    
    return 0;
}