#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define LOG_TAG "onnxNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)


cv::Mat depthMapToPointCloud(cv::Mat depthMap, std::string& pointString);

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_onnx_MainActivity_nativeProcess(JNIEnv *env, jobject thiz,jobjectArray arr, jint height, jint width) {
    //create a float 2D mat with the given height and width
    cv::Mat mat(height, width, CV_32FC1);
    //fill the mat with arr values
    for (int i = 0; i < height; i++) {
        jfloatArray row = (jfloatArray) env->GetObjectArrayElement(arr, i);
        jfloat *rowElements = env->GetFloatArrayElements(row, 0);
        for (int j = 0; j < width; j++) {
            mat.at<float>(i, j) = rowElements[j];
        }
        env->ReleaseFloatArrayElements(row, rowElements, 0);
        env->DeleteLocalRef(row);
    }

    // Padding information: [top, bottom, left, right]
    // padInfo = {0, 0, 121, 122};

    int paddingTop = 0;
    int paddingBottom = 0;
    int paddingLeft = 121;
    int paddingRight = 122;

    //depth = mat[
    //        paddingTop : input_size[0] - paddingBottom,
    //        paddingLeft : input_size[1] - paddingRight,
    //    ]
    //get above depth mat
    cv::Mat depth = mat(cv::Range(paddingTop, mat.rows - paddingBottom),
                        cv::Range(paddingLeft, mat.cols - paddingRight));

    cv::Mat resizedDepth;
    int resizedHeight = 480;
    int resizedWidth = 640;
    cv::resize(depth, resizedDepth, cv::Size(resizedWidth, resizedHeight), 0, 0, cv::INTER_LINEAR);

    //string to convert point cloud to string
    std::string pointString="";

    //convert depth map to point cloud
    cv::Mat pointCloud = depthMapToPointCloud(resizedDepth, pointString);

   //return pointString as jstring
    return env->NewStringUTF(pointString.c_str());
}

cv::Mat depthMapToPointCloud(cv::Mat depthMap, std::string& pointString) {
    int pointCloudWidth = depthMap.cols;
    int pointCloudHeigth = depthMap.rows;

    float fx = 488.3022098;
    float fy = 488.3022098;
    float ppx = 320.99165;
    float ppy = 235.4008464;
    float depthScalar = 1.0;

    //create a mat with 3 channels to hold the point cloud with sizes of pointCloudWidth and pointCloudHeigth
    cv::Mat pointCloud(pointCloudHeigth, pointCloudWidth, CV_32FC3);
    std::ostringstream oss;

    //fill the point cloud mat with the depth map values
    for (int v = 0; v < pointCloudHeigth; ++v)
    {
        for (int u = 0; u < pointCloudWidth; ++u)
        {
            float z = depthMap.at<float>(v, u) * depthScalar;
            float x = (static_cast<float>(u) - ppx) * z / fx;
            float y = (static_cast<float>(v) - ppy) * z / fy;
            pointCloud.at<cv::Vec3f>(v, u) = cv::Vec3f(x, y, z);

            oss << x << "," << y << "," << z << "\n";
        }
    }



    //remove last comma in pointString
    pointString = oss.str();
    if (!pointString.empty()) {
        pointString.pop_back();
    }

    return pointCloud;
}
