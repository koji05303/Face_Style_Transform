#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
    cv::Mat input_image = imread("datasets_sketch/output.jpg");

    if (input_image.empty())
    {
        cout << "image not found" << endl;
        return 1;
    }

    cv::Mat gray_image;
    cv::Mat blur_image;
    cv::Mat blend_image;
    cv::Mat clahe_image;
    cv::Mat sketch_image;

    // Convert input image to grayscale
    cv::cvtColor(input_image, gray_image, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur to the grayscale image
    cv::GaussianBlur(gray_image, blur_image, cv::Size(21, 21), 0, 0);

    // divide the blurred image by the grayscale image
    cv::divide(gray_image, blur_image, blend_image, 256);

    // Create a CLAHE object
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(6);

    // Apply CLAHE to the blended image
    clahe->apply(blend_image, clahe_image);

    // Display the result
    cv::imwrite("saved_images/sketch/sketch.jpg", clahe_image);
    cv::waitKey(0);

    return 0;
}
