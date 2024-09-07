#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

using namespace std;
using namespace cv;
using namespace dlib;

int main() {
    // 輸入影像
    Mat image = imread("input_image/input.jpg");
 
    // import Haarcascades
    CascadeClassifier face_cascade;
    face_cascade.load("haarcascade_frontalface_default.xml");
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(image, faces, 1.1, 3, 0, cv::Size(30, 30));

    // 找到人臉的中心點
    Point center(faces[0].x + faces[0].width / 2,
        faces[0].y + faces[0].height / 2);

    // 計算裁剪框的大小
    int width = 400;
    int height = 450;
    Size crop_size(width, height);

    // 裁剪影像
    Mat crop = image(Rect(center.x - crop_size.width / 2,
        center.y - crop_size.height / 2,
        crop_size.width, crop_size.height));

    // 臉部座標輸出
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
    dlib::cv_image<dlib::bgr_pixel> img(crop);
   std:: vector<dlib::rectangle>dets = detector(img);
   full_object_detection shape = pose_model(img, dets[0]);

   std::ofstream outfile_aging("datasets_old/output.jpg.txt");
   std::ofstream outfile_baby("datasets_baby/output.jpg.txt");
   for (unsigned long i = 0; i < shape.num_parts(); i++)
   {
       outfile_aging << shape.part(i).x() << " " << shape.part(i).y() << std::endl;
   }
   outfile_aging.close();
   for (unsigned long j = 0; j < shape.num_parts(); j++)
   {
       outfile_baby << shape.part(j).x() << " " << shape.part(j).y() << std::endl;
   }
   outfile_baby.close();

    // 顯示原始影像和裁剪後的影像
    imwrite("datasets_sketch/output.jpg", crop);
    imwrite("datasets_old/output.jpg", crop);
    imwrite("datasets_baby/output.jpg", crop);

    waitKey(0);

    return 0;
}