#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


// Function to resize and crop image to target size while maintaining aspect ratio
Mat resizeAndCrop(const Mat& src, const Size& targetSize) {
    Mat dst;
    double aspectSrc = static_cast<double>(src.cols) / src.rows;
    double aspectTarget = static_cast<double>(targetSize.width) / targetSize.height;

    int newWidth, newHeight;
    if (aspectSrc > aspectTarget) {
        newWidth = targetSize.width;
        newHeight = static_cast<int>(targetSize.width / aspectSrc);
    }
    else {
        newHeight = targetSize.height;
        newWidth = static_cast<int>(targetSize.height * aspectSrc);
    }

    resize(src, dst, Size(newWidth, newHeight));

    // Ensure the new dimensions are within the bounds
    if (newWidth > targetSize.width) {
        newWidth = targetSize.width;
    }
    if (newHeight > targetSize.height) {
        newHeight = targetSize.height;
    }

    // Calculate the center crop region
    int x = max((dst.cols - targetSize.width) / 2, 0);
    int y = max((dst.rows - targetSize.height) / 2, 0);

    Rect cropRegion(x, y, min(targetSize.width, dst.cols), min(targetSize.height, dst.rows));
    dst = dst(cropRegion);

    return dst;
}

int main()
{
    //import image
    cv::Mat img_original = cv::imread("saved_images/original/original.jpg");
    cv::Mat img_sketch = cv::imread("saved_images/sketch/sketch.jpg");
    cv::Mat img_baby = cv::imread("saved_images/baby/baby.jpg");
    cv::Mat img_aging = cv::imread("saved_images/aging/aging.jpg");
    cv::Mat img_funhouse = cv::imread("saved_images/funhouse/funhouse.jpg");
    cv::Mat img_cartoon = cv::imread("saved_images/cartoon/cartoon.jpg");
    
    //resize each image to 400 x 450
    Size targetSize(400, 450);
    cv::resize(img_original, img_original, cv::Size(400, 450));
    cv::resize(img_sketch, img_sketch, cv::Size(400, 450));
    cv::resize(img_funhouse, img_funhouse, cv::Size(400, 450));
    cv::resize(img_cartoon, img_cartoon, cv::Size(400, 450));
    img_baby = resizeAndCrop(img_baby, targetSize);
    img_aging = resizeAndCrop(img_aging, targetSize);

    //cv::resize(img_baby, img_baby, cv::Size(400, 450));
    //cv::resize(img_aging, img_aging, cv::Size(400, 450));

    // Concatenate six images into one image with 2x3 grid using OpenCV hconcat function
    // First row: original, sketch, baby
    cv::Mat img_hconcat_1, img_hconcat_2;
    cv::hconcat(img_original, img_sketch, img_hconcat_1);
    cv::hconcat(img_hconcat_1, img_baby, img_hconcat_1);

    // Second row: aging, funhouse, cartoon
    cv::hconcat(img_aging, img_funhouse, img_hconcat_2);
    cv::hconcat(img_hconcat_2, img_cartoon, img_hconcat_2);

    // Concatenate two rows
    cv::Mat img_vconcat;
    cv::vconcat(img_hconcat_1, img_hconcat_2, img_vconcat);

    // Resize final image to fit the display window if needed
    // cv::resize(img_vconcat, img_vconcat, cv::Size(1200, 900));

    //cv::imshow("original", img_original);
    //cv::imshow("sketch", img_sketch);
    //cv::imshow("baby", img_baby);
    //cv::imshow("aging", img_aging);
    //cv::imshow("funhouse", img_funhouse);
    //cv::imshow("cartoon", img_cartoon);
    //cv::imshow("hconcat1", img_hconcat_1);
    //cv::imshow("hconcat2", img_hconcat_2); 
    //cv::imshow("concat", img_vconcat);

    cv::imwrite("saved_images/result_3x2/result.jpg", img_vconcat);

    cv::waitKey(0);
    return 0;
}
