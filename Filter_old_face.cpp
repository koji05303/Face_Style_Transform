#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>

#define M_PI       3.1415926535897932384626433832795028841971693993

using namespace cv;
using namespace std;

void similarityTransform(std::vector<cv::Point2f>& inPoints, std::vector<cv::Point2f>& outPoints, cv::Mat& tform)
{
    float s60 = sin(60 * M_PI / 180.0);
    float c60 = cos(60 * M_PI / 180.0);

    vector <Point2f> inPts = inPoints;
    vector <Point2f> outPts = outPoints;

    inPts.push_back(cv::Point2f(0, 0));
    outPts.push_back(cv::Point2f(0, 0));

    inPts[2].x = c60 * (inPts[0].x - inPts[1].x) - s60 * (inPts[0].y - inPts[1].y) + inPts[1].x; // x' = cos(theta) * (x - x0) - sin(theta) * (y - y0) + x0
    inPts[2].y = s60 * (inPts[0].x - inPts[1].x) + c60 * (inPts[0].y - inPts[1].y) + inPts[1].y; // y' = sin(theta) * (x - x0) + cos(theta) * (y - y0) + y0

    outPts[2].x = c60 * (outPts[0].x - outPts[1].x) - s60 * (outPts[0].y - outPts[1].y) + outPts[1].x; // x' = cos(theta) * (x - x0) - sin(theta) * (y - y0) + x0
    outPts[2].y = s60 * (outPts[0].x - outPts[1].x) + c60 * (outPts[0].y - outPts[1].y) + outPts[1].y; // y' = sin(theta) * (x - x0) + cos(theta) * (y - y0) + y0

    tform = cv::estimateAffinePartial2D(inPts, outPts); // Compute an optimal affine transformation between two 2D point sets
}

// Read points from list of text file
void readPoints(vector<string> pointsFileNames, vector<vector<Point2f> >& pointsVec) {

    for (size_t i = 0; i < pointsFileNames.size(); i++)
    {
        vector<Point2f> points;
        ifstream ifs(pointsFileNames[i]);
        float x, y;
        while (ifs >> x >> y)
            points.push_back(Point2f((float)x, (float)y));

        pointsVec.push_back(points);
    }

}

// Read names from the directory
void readFileNames(string dirName, vector<string>& imageFnames, vector<string>& ptsFnames)
{
    DIR* dir;
    struct dirent* ent;
    int count = 0;

    //image extensions
    string imgExt = "jpg";
    string txtExt = "txt";
    vector<string> files;

    if ((dir = opendir(dirName.c_str())) != NULL)
    {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL)
        {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            {
                //count++;
                continue;
            }
            string temp_name = ent->d_name;
            files.push_back(temp_name);
        }
        std::sort(files.begin(), files.end());
        for (int it = 0; it < files.size(); it++)
        {
            string path = dirName;
            string fname = files[it];
            if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos)
            {
                path.append(fname);
                imageFnames.push_back(path);
            }
            else if (fname.find(txtExt, (fname.length() - txtExt.length())) != std::string::npos)
            {
                path.append(fname);
                ptsFnames.push_back(path);
            }
        }
        closedir(dir);
    }
}

// Calculate Delaunay triangles for set of points// Returns the vector of indices of 3 points for each triangle
static void calculateDelaunayTriangles(Rect rect, vector<Point2f>& points, vector< vector<int> >& delaunayTri) {

    // Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);

    // Insert points into subdiv
    for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
        subdiv.insert(*it);

    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point2f> pt(3);
    vector<int> ind(3);

    for (size_t i = 0; i < triangleList.size(); i++)
    {
        Vec6f t = triangleList[i];
        pt[0] = Point2f(t[0], t[1]);
        pt[1] = Point2f(t[2], t[3]);
        pt[2] = Point2f(t[4], t[5]);

        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
            for (int j = 0; j < 3; j++)
                for (size_t k = 0; k < points.size(); k++)
                    if (abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
                        ind[j] = k;

            delaunayTri.push_back(ind);
        }
    }

}

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat& warpImage, Mat& src, vector<Point2f>& srcTri, vector<Point2f>& dstTri)
{
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform(srcTri, dstTri);

    // Apply the Affine Transform just found to the src image
    warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void warpTriangle(Mat& img1, Mat& img2, vector<Point2f> t1, vector<Point2f> t2)
{
    double alpha = 0.45;
//    img1 = img1 * alpha;
    // Find bounding rectangle for each triangle
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);

    // Offset points by left top corner of the respective rectangles
    vector<Point2f> t1Rect, t2Rect;
    vector<Point> t2RectInt;
    for (int i = 0; i < 3; i++)
    {
        t2RectInt.push_back(Point((int)(t2[i].x - r2.x), (int)(t2[i].y - r2.y)));

        t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
        t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
    }

    // Get mask by filling triangle
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

    // Apply warpImage to small rectangular patches
    Mat img1Rect, img2Rect;
    img1(r1).copyTo(img1Rect);

    Mat warpImage = Mat::zeros(r2.height, r2.width, img1Rect.type());

    applyAffineTransform(warpImage, img1Rect, t1Rect, t2Rect);

    // Copy triangular region of the rectangular patch to the output image
    warpImage = warpImage * alpha;
    multiply(warpImage, mask, warpImage);
    multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + warpImage;

   //debug
   //imshow("mask",mask);
   //imshow("warp",warpImage); 
    //cout << "img2(r2) type: " << img2(r2).type() << ", channels: " << img2(r2).channels() << endl;
    //cout << "mask type: " << mask.type() << ", channels: " << mask.channels() << endl;
}

// Constrains points to be inside boundary
void constrainPoint(Point2f& p, Size sz)
{
    p.x = min(max((double)p.x, 0.0), (double)(sz.width - 1));
    p.y = min(max((double)p.y, 0.0), (double)(sz.height - 1));

}

int main(int argc, char** argv)
{
    // Directory containing images.
    string dirName = "datasets_old";

    // Add slash to directory name if missing
    if (!dirName.empty() && dirName.back() != '/')
        dirName += '/';

    // Dimensions of output image
    int w = 450;
    int h = 600;

    // Read images in the directory
    vector<string> imageNames, ptsNames;
    readFileNames(dirName, imageNames, ptsNames);

    //cout << imageNames.size() << ptsNames.size();

    // Exit program if no images or pts are found or if the number of image files does not match with the number of point files
    if (imageNames.empty() || ptsNames.empty() || imageNames.size() != ptsNames.size()) {
        exit(EXIT_FAILURE);
    }

    // Read points
    vector<vector<Point2f> > allPoints;
    readPoints(ptsNames, allPoints);

    int n = allPoints[0].size();
    //cout << n<< endl;

    // Read images
    vector<Mat> images;
    for (size_t i = 0; i < imageNames.size(); i++)
    {
        Mat img = imread(imageNames[i]);

        img.convertTo(img, CV_32FC3, 1 / 255.0);

        if (!img.data)
        {
            cout << "image " << imageNames[i] << " not read properly" << endl;
        }
        else
        {
            images.push_back(img);
        }
    }

    if (images.empty())
    {
        cout << "No images found " << endl;
        exit(EXIT_FAILURE);
    }

    int numImages = images.size();

    // Eye corners
    vector<Point2f> eyecornerDst, eyecornerSrc;
    eyecornerDst.push_back(Point2f(0.3 * w, h / 3));
    eyecornerDst.push_back(Point2f(0.7 * w, h / 3));

    eyecornerSrc.push_back(Point2f(0, 0));
    eyecornerSrc.push_back(Point2f(0, 0));

    // Space for normalized images and points.
    vector <Mat> imagesNorm;
    vector < vector <Point2f> > pointsNorm;

    // Space for average landmark points
    vector <Point2f> pointsAvg(allPoints[0].size());

    // 8 Boundary points for Delaunay Triangulation
    vector <Point2f> boundaryPts;
    boundaryPts.push_back(Point2f(0, 0));
    boundaryPts.push_back(Point2f(w / 2, 0));
    boundaryPts.push_back(Point2f(w - 1, 0));
    boundaryPts.push_back(Point2f(w - 1, h / 2));
    boundaryPts.push_back(Point2f(w - 1, h - 1));
    boundaryPts.push_back(Point2f(w / 2, h - 1));
    boundaryPts.push_back(Point2f(0, h - 1));
    boundaryPts.push_back(Point2f(0, h / 2));

    // Warp images and trasnform landmarks to output coordinate system,
    // and find average of transformed landmarks.

    for (size_t i = 0; i < images.size(); i++)
    {

        vector <Point2f> points = allPoints[i];

        // The corners of the eyes are the landmarks number 36 and 45
        eyecornerSrc[0] = allPoints[i][36];
        eyecornerSrc[1] = allPoints[i][45];

        // Calculate similarity transform
        Mat tform;
        similarityTransform(eyecornerSrc, eyecornerDst, tform);

        // Apply similarity transform to input image and landmarks
        Mat img = Mat::zeros(h, w, CV_32FC3);
        warpAffine(images[i], img, tform, img.size());
        transform(points, points, tform);

        // Calculate average landmark locations
        for (size_t j = 0; j < points.size(); j++)
        {
            pointsAvg[j] += points[j] * (1.0 / numImages);
        }

        // Append boundary points. Will be used in Delaunay Triangulation
        for (size_t j = 0; j < boundaryPts.size(); j++)
        {
            points.push_back(boundaryPts[j]);
        }

        pointsNorm.push_back(points);
        imagesNorm.push_back(img);
    }

    // Append boundary points to average points.
    for (size_t j = 0; j < boundaryPts.size(); j++)
    {
        pointsAvg.push_back(boundaryPts[j]);
    }

    // Calculate Delaunay triangles
    Rect rect(0, 0, w, h);
    vector< vector<int> > dt;
    calculateDelaunayTriangles(rect, pointsAvg, dt);

    // Space for output image
    Mat output = Mat::zeros(h, w, CV_32FC3);
    Size size(w, h);

    // Warp input images to average image landmarks

    for (size_t i = 0; i < numImages; i++)
    {
        Mat img = Mat::zeros(h, w, CV_32FC3);
        // Transform triangles one by one
        for (size_t j = 0; j < dt.size(); j++)
        {
            // Input and output points corresponding to jth triangle
            vector<Point2f> tin, tout;
            for (int k = 0; k < 3; k++)
            {
                Point2f pIn = pointsNorm[i][dt[j][k]];
                constrainPoint(pIn, size);

                Point2f pOut = pointsAvg[dt[j][k]];
                constrainPoint(pOut, size);

                tin.push_back(pIn);
                tout.push_back(pOut);
            }

            warpTriangle(imagesNorm[i], img, tin, tout);
        }

        // Now, add the image intensities for averaging
        output = output + img;
    }


    // Divide by numImages to get average
    double alpha = 0.9;
    output = output * alpha;
    Mat output_raw = output.clone();

    Mat output_8UC3;
    output.convertTo(output_8UC3, CV_8UC3, 255.0);

    //Try hist & clahe
    Mat labimage, claheimage;
    cvtColor(output_8UC3, labimage, COLOR_BGR2Lab);
    cvtColor(output_8UC3, claheimage, COLOR_BGR2Lab);

    vector<Mat> labchannel(3);
    vector<Mat> clahechannel(3);

    split(labimage, labchannel);
    split(claheimage, clahechannel);

    //hist
    equalizeHist(labchannel[0], labchannel[0]);

    //clahe
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(1.4); // 調整參數以控制對比度
    clahe->apply(clahechannel[0], clahechannel[0]);

    //hist merge
    Mat equalizedlabimage;
    merge(labchannel, equalizedlabimage);
    Mat equalizedOutput;
    cvtColor(equalizedlabimage, equalizedOutput, COLOR_Lab2BGR);

    //clahe merge
    Mat clahelabimage;
    merge(clahechannel, claheimage);
    cvtColor(claheimage, clahelabimage, COLOR_Lab2BGR);



    //imshow("CLAHE", clahelabimage);
    imwrite("saved_images/aging/aging.jpg",clahelabimage);

    waitKey(0);

    return 0;
}