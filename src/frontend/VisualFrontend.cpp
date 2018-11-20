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
		track(im_gray, trackedPoints);
		auto feend = chrono::steady_clock::now();
		cout << "klt running time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;


		//Save tracked points
		oldPoints = trackedPoints;
	}

	//Extract new points
	auto festart = chrono::steady_clock::now();
	extract(im_gray, newPoints);
	auto feend = chrono::steady_clock::now();
	cout << "new feature time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;

	//save old image
	im_prev = im_gray;

}

void VisualFrontend::extract1(Mat& im_gray, Features2D& newPoints)
{
    // fill the blank
}

void VisualFrontend::track1(Mat& im_gray, Features2D& trackedPoints)
{
    // fill the blank
}
