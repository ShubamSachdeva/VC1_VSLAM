#include "frontend/OccupancyGrid.h"
using namespace std;
using namespace cv;

OccupancyGrid::OccupancyGrid()
{
	//initializer();
	initializer1();
}

void OccupancyGrid::initializer1()
{
	Ix = 16;
	Iy = 16;
	resetGrid1();

}

void OccupancyGrid::setImageSize1(size_t cols, size_t rows)
{
	Ix = cols / nx;
	Iy = rows / ny;

}

void OccupancyGrid::addPoint1(Point2f& p)
{
	size_t x = p.x/Ix;
	size_t y = p.y/Iy;
	if(x >= nx || y >= ny)
        	return;

    	isFree[x][y] = false;
}

bool OccupancyGrid::isNewFeature1(Point2f& p)
{
	int x = p.x/Ix;
	int y = p.y/Iy;
	bool nothingClose = true;

    for(unsigned int i = std::max(0,x-1); i < std::min((int)nx, x+2); i++)
        for(unsigned int j = std::max(0, y-1); j < std::min((int)ny, y+2); j++)
            nothingClose = nothingClose && isFree[i][j];

    return nothingClose;
}

void OccupancyGrid::resetGrid1()
{
	for(int i=0; i<nx; i++)
		for(int j=0; j<ny; j++)
			isFree[i][j] = true;
	  
}

