package com.example.onnx;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OrtException;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Camera2App";
    private TextureView textureView;
    private ImageView imageView;
    private TextView textView;
    private CameraDevice cameraDevice;
    private CameraCaptureSession cameraCaptureSession;
    private CaptureRequest.Builder captureRequestBuilder;

    private ImageReader imageReader;
    private Button openCameraButton;
    private Button closeCameraButton;
    private Button processCameraButton;
    private static final int REQUEST_WRITE_STORAGE = 112;
    Context context;

    private volatile boolean shouldProcessFrame = false;

    onnxModelHandler.OnnxModelHandler onnxModelHandler= null;

    // This will be our dedicated background thread for processing images.
    private ExecutorService processingExecutor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        context= getApplicationContext();

        // Initialize the single-thread executor.
        processingExecutor = Executors.newSingleThreadExecutor();


        if (ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions((Activity) context,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_WRITE_STORAGE);
        }

        textureView = findViewById(R.id.textureView);
        openCameraButton = findViewById(R.id.openCameraButton);
        closeCameraButton = findViewById(R.id.closeCameraButton);
        processCameraButton=findViewById(R.id.processCameraButton);
        imageView = findViewById(R.id.processedImageView);
        textView = findViewById(R.id.infoTextView);

        openCameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openCameraButton.setEnabled(false);
                closeCameraButton.setEnabled(true);
                processCameraButton.setEnabled(true);
                openCamera();
            }
        });

        closeCameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                closeCameraButton.setEnabled(false);
                processCameraButton.setEnabled(false);
                closeCamera();
                openCameraButton.setEnabled(true);
            }
        });

        processCameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(TAG, "Process button clicked");
                shouldProcessFrame=true;
            }
        });

        initOnnx();

    }


    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            cameraDevice = camera;
            startCameraPreview();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            cameraDevice.close();
        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            cameraDevice.close();
            cameraDevice = null;
        }
    };

    private void openCamera() {
        runOnUiThread(() -> {
            textureView.setVisibility(View.VISIBLE); // Hide texture view
            imageView.setVisibility(View.VISIBLE);
        });

        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            String cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                return;
            }
            manager.openCamera(cameraId, stateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void closeCamera() {
        Log.d(TAG, "closeCamera called");
        try {
            if (cameraCaptureSession != null) {
                cameraCaptureSession.stopRepeating(); // Stop repeating requests
                cameraCaptureSession.abortCaptures(); // Abort any pending captures
                cameraCaptureSession.close();
                cameraCaptureSession = null;
                Log.d(TAG, "CameraCaptureSession closed");
            }
        } catch (CameraAccessException e) {
            Log.e(TAG, "Error stopping/aborting session during close", e);
        } catch (IllegalStateException e){
            Log.e(TAG, "Illegal state during session stop/abort (already closed?)", e);
        }

        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
            Log.d(TAG, "CameraDevice closed");
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
            Log.d(TAG, "ImageReader closed");
        }
        // Update button states on UI thread
        runOnUiThread(() -> {
            openCameraButton.setEnabled(true);
            closeCameraButton.setEnabled(false);
            textureView.setVisibility(View.INVISIBLE); // Hide texture view
            imageView.setVisibility(View.INVISIBLE);
        });
        Log.i(TAG, "Camera closed and resources released.");

    }

    private void startCameraPreview() {
        SurfaceTexture texture = textureView.getSurfaceTexture();
        assert texture != null;
        texture.setDefaultBufferSize(640, 480);
        Surface surface = new Surface(texture);

        setupImageReader();

        try {
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);

            // Add the ImageReader surface as a target
            Surface imageReaderSurface = imageReader.getSurface();
            captureRequestBuilder.addTarget(imageReaderSurface);

            // Create the capture session with both surfaces
            cameraDevice.createCaptureSession(Arrays.asList(surface, imageReaderSurface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            cameraCaptureSession = session;
                            updatePreview();
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            Toast.makeText(MainActivity.this, "Failed to configure camera", Toast.LENGTH_SHORT).show();
                        }
                    }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void updatePreview() {
        if (cameraDevice == null) {
            return;
        }
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        try {
            cameraCaptureSession.setRepeatingRequest(captureRequestBuilder.build(), null, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void setupImageReader() {
        imageReader = ImageReader.newInstance(640, 480, ImageFormat.JPEG, 2);
        imageReader.setOnImageAvailableListener(reader -> {
            Image image = reader.acquireLatestImage();

            if (image != null) {

                Log.d(TAG, "setupImageReader: " );

                Bitmap bitmap = imageToBitmap(image);
                image.close();

                if (bitmap != null && shouldProcessFrame) {
                    //run on ui thread hide process button
                    runOnUiThread(() -> processCameraButton.setVisibility(View.INVISIBLE));

                    processingExecutor.submit(() -> {
                        //i want to get time difference for inference
                        long startTime = System.currentTimeMillis();

                        //onnx inference
                        if (onnxModelHandler != null) {
                            try {
                                float[] output = onnxModelHandler.runInference(bitmap);
                                // Process the output as needed
                                Log.d(TAG, "setupImageReader: onnx inference success");

                            } catch (OrtException e) {
                                Log.d(TAG, "setupImageReader: onnx inference faield "+ e);
                            }
                        }else
                            Log.d(TAG, "setupImageReader: onnx model null ");


                        long endTime = System.currentTimeMillis();
                        double inferenceTime = (double) (endTime - startTime) /1000;

                        shouldProcessFrame=false;
                        runOnUiThread(() -> {
                            processCameraButton.setVisibility(View.VISIBLE);
                            textView.setText("Last Inference Time " + inferenceTime + " ms");
                        });

                    });

                }
            }
        }, null);
    }

    private Bitmap imageToBitmap(Image image) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
    }

    private void initOnnx(){
        try {
            onnxModelHandler = new onnxModelHandler.OnnxModelHandler(this, "metric3d_vit_small.onnx");
            Toast.makeText(this, "onnx model loaded", Toast.LENGTH_SHORT).show();
        } catch (OrtException | IOException e) {
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }


}