#include "frontend/VisualFrontend.h"
#include <chrono>
using namespace std;
using namespace cv;

void VisualFrontend::downloadpts(const GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

void VisualFrontend::downloadmask(const GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

VisualFrontend::VisualFrontend()
{
	//Initialise detector
//	std::string detectorType = "Feature2D.BRISK";
//
//	detector = Algorithm::create<FeatureDetector>(detectorType);
//	detector->set("thres", thresholdExtraction);
	//Initialize ID
    gpu_detector = GoodFeaturesToTrackDetector_GPU(250, 0.01, 0);

    d_pyrLK.winSize.width = 21;
    d_pyrLK.winSize.height = 21;
    d_pyrLK.maxLevel = 3;
    d_pyrLK.iters = 30;

	newId = 0;

}

void VisualFrontend::trackAndExtract(cv::Mat& im_gray, Features2D& trackedPoints, Features2D& newPoints)
{
	if (oldPoints.size() > 0)
	{
        //Track prevoius points with optical flow
		auto festart = chrono::steady_clock::now();
		
		track1(im_gray, trackedPoints);
		auto feend = chrono::steady_clock::now();
		cout << "klt running time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;


		//Save tracked points
		oldPoints = trackedPoints;
	}
	
	//Extract new points
	auto festart = chrono::steady_clock::now();
	extract1(im_gray, newPoints);
	auto feend = chrono::steady_clock::now();
	cout << "new feature time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;

	//save old image
	im_prev = im_gray;

}

void VisualFrontend::extract1(Mat& im_gray, Features2D& newPoints)
{
	GpuMat img(im_gray);
	GpuMat new_points;
	gpu_detector(img, new_points);
	vector<Point2f> new_points_v;

	downloadpts(new_points, new_points_v);
	grid.setImageSize1(im_gray.cols, im_gray.rows);
	
	for(auto points: new_points_v)
	{
		if(grid.isNewFeature1(points))
		{
			newPoints.addPoint(points, newId);
			oldPoints.addPoint(points, newId);
			newId++;
		}
			
	}
	grid.resetGrid1();
	
}

void VisualFrontend::track1(Mat& im_gray, Features2D& trackedPoints)
{
    // fill the blank
	GpuMat img1(im_prev);
	GpuMat img2(im_gray);
	auto points = oldPoints.getPoints();
	Mat previous_points = Mat(1,points.size(), CV_32FC2, points.data()).clone();
	GpuMat previous_points_g(previous_points);
	GpuMat next_points_g;
	GpuMat status_g, status1_g;
	GpuMat previous_points_bk_g;
	
	d_pyrLK.sparse(img1, img2, previous_points_g, next_points_g, status_g);
	d_pyrLK.sparse(img2, img1, next_points_g, previous_points_bk_g, status1_g);
	
 	vector<Point2f> points_bk, next_points;
    	downloadpts(previous_points_bk_g, points_bk);
	downloadpts(next_points_g, next_points);

	vector<uchar> status, status1;
	downloadmask(status_g, status);
	downloadmask(status1_g, status1);
	
	 for (size_t i = 0; i < status.size(); i++)
        	status[i] = ((norm(points_bk[i] - points[i]) <= thresholdFBError) && status[i] && status1[i]);
	
    	trackedPoints = Features2D(oldPoints, next_points, status);
	for (Point2f& point : trackedPoints)
	{
				
		grid.addPoint1(point);
	}

	
	
	

}
