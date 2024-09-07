#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Function to apply sine wave distortion effect
Mat applyWaveEffect(const Mat& src) {
    Mat dst = src.clone();
    int rows = src.rows;
    int cols = src.cols;

    // Parameters for the wave effect
    double amplitude =8.0; // Wave amplitude
    double frequency = 0.02; // Wave frequency

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            // Calculate the wave distortion
            //int newX = x + static_cast<int>(amplitude * sin(2 * CV_PI * y * frequency));
            int newX = x;
            int newY = y + static_cast<int>(amplitude * cos(2 * CV_PI * x * frequency));

            // Ensure the new coordinates are within image bounds
            newX = min(max(newX, 0), cols - 1);
            newY = min(max(newY, 0), rows - 1);

            dst.at<Vec3b>(y, x) = src.at<Vec3b>(newY, newX);
        }
    }
    return dst;
}

//Function to apply cartoon effects
Mat applyCartoonEffect(const Mat& src)
{
    Mat imgColor, imgGray, imgEdges;

    // Step 1: Reduce the number of colors in the image
    pyrMeanShiftFiltering(src, imgColor, 21, 51);

    // Step 2: Convert to grayscale
    cvtColor(src, imgGray, COLOR_BGR2GRAY);

    // Step 3: Apply median blur to remove noise
    medianBlur(imgGray, imgGray, 7);

    // Step 4: Detect edges using adaptive thresholding
    adaptiveThreshold(imgGray, imgEdges, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 2);

    // Step 5: Convert edges back to color image
    cvtColor(imgEdges, imgEdges, COLOR_GRAY2BGR);

    // Step 6: Combine color image with edges
    Mat cartoon = imgColor & imgEdges;

    return cartoon;
}

int main() {
    // Load the image
    Mat src = imread("datasets_funhouse_cartoon/output.jpg");
    // Check if image is loaded successfully
    if (src.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Apply wave effect
    Mat dst = applyWaveEffect(src);

    // Apply cartoon effect
    Mat dst_c = applyCartoonEffect(src);

    // Display the original and distorted images
    //imshow("Original Image", src);
    //imshow("Wave Effect", dst);
    //imshow("Cartoon Effect", dst_c);

    // Save the distorted image
    cv::imwrite("saved_images/funhouse/funhouse.jpg", dst);
    cv::imwrite("saved_images/cartoon/cartoon.jpg", dst_c);

    waitKey(0);

    return 0;
}
