#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define LOG_TAG "onnxNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)


extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_onnx_MainActivity_getString(JNIEnv *env, jobject thiz,jobjectArray arr, jint height, jint width) {
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
    int paddingLeft = 1;
    int paddingRight = 1;

    //depth = mat[
    //        paddingTop : input_size[0] - paddingBottom,
    //        paddingLeft : input_size[1] - paddingRight,
    //    ]
    //get above depth mat
    cv::Mat depth = mat(cv::Range(paddingTop, mat.rows - paddingBottom),
                        cv::Range(paddingLeft, mat.cols - paddingRight));


    //convert depth to java array and return it to java
    jobjectArray depthArray = env->NewObjectArray(depth.rows, env->FindClass("[F"), nullptr);
    for (int i = 0; i < depth.rows; i++) {
        jfloatArray row = env->NewFloatArray(depth.cols);
        env->SetFloatArrayRegion(row, 0, depth.cols, depth.ptr<float>(i));
        env->SetObjectArrayElement(depthArray, i, row);
        env->DeleteLocalRef(row);
    }

    //retrn depth array
    return depthArray;
}
