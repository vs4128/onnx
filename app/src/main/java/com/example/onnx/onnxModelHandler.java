package com.example.onnx;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import android.content.Context;
import android.graphics.Bitmap;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
public class onnxModelHandler {

    public static class OnnxModelHandler {
        private OrtEnvironment ortEnvironment;
        private OrtSession ortSession;

        // Utility to copy asset to internal storage
        private String copyAssetToFile(Context context, String assetName) throws IOException {
            File outFile = new File(context.getFilesDir(), assetName);
            if (!outFile.exists()) {
                try (InputStream in = context.getAssets().open(assetName);
                     FileOutputStream out = new FileOutputStream(outFile)) {
                    byte[] buffer = new byte[4096];
                    int read;
                    while ((read = in.read(buffer)) != -1) {
                        out.write(buffer, 0, read);
                    }
                }
            }
            return outFile.getAbsolutePath();
        }

        // Updated constructor
        public OnnxModelHandler(Context context, String assetName) throws OrtException, IOException {
            ortEnvironment = OrtEnvironment.getEnvironment();
            String modelPath = copyAssetToFile(context, assetName);
            ortSession = ortEnvironment.createSession(modelPath, new OrtSession.SessionOptions());
        }

        public float[] runInference(Bitmap bitmap) throws OrtException {
            float[] inputTensor = prepareInput(bitmap);
            OnnxTensor input = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(inputTensor), new long[]{1, 3, 616, 1064});

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("pixel_values", input);
            OrtSession.Result results = ortSession.run(inputs);

            return flattenOutput((float[][][]) results.get(0).getValue());
        }

        private float[] prepareInput(Bitmap bitmap) {
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 1064, 616, true);
            int width = resizedBitmap.getWidth();
            int height = resizedBitmap.getHeight();
            float[] inputData = new float[3 * width * height];

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pixel = resizedBitmap.getPixel(x, y);
                    int index = y * width + x;
                    inputData[index] = ((pixel >> 16) & 0xFF) / 255.0f; // Red
                    inputData[width * height + index] = ((pixel >> 8) & 0xFF) / 255.0f; // Green
                    inputData[2 * width * height + index] = (pixel & 0xFF) / 255.0f; // Blue
                }
            }
            return inputData;
        }

        private float[] flattenOutput(float[][][] output) {
            int d1 = output.length;
            int d2 = output[0].length;
            int d3 = output[0][0].length;
            float[] flat = new float[d1 * d2 * d3];
            int idx = 0;
            for (int i = 0; i < d1; i++)
                for (int j = 0; j < d2; j++)
                    for (int k = 0; k < d3; k++)
                        flat[idx++] = output[i][j][k];
            return flat;
        }
    }
}
