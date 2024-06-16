package com.example.exercise3;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.nio.ByteOrder;

import com.example.exercise3.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.metadata.schema.ImageSize;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {


    private TextView txtClass;
    private ImageView imgFruit;
    private Button btnGallery, btnPicture;

    private int ImageSize = 32;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        txtClass = (TextView) findViewById(R.id.txtClass);
        imgFruit = (ImageView) findViewById(R.id.imgFruit);
        btnPicture = (Button) findViewById(R.id.btnPicture);
        btnGallery = (Button) findViewById(R.id.btnPicture);

        btnPicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                }else{
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100 );
                }
            }
        });


    }
    public void viewGallery(View v){
         Intent intentGallery = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
         startActivityForResult(intentGallery, 2);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        if (resultCode==RESULT_OK) {
            if (requestCode==1) {
                Bitmap photo = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(photo.getWidth(), photo.getHeight());
                photo = ThumbnailUtils.extractThumbnail(photo, dimension, dimension);
                imgFruit.setImageBitmap(photo);
                photo = Bitmap.createScaledBitmap(photo, ImageSize, ImageSize, false);
                classifyImg(photo);

            } else if (requestCode == 2) {

                Uri dataUri = data.getData();
                Bitmap photo = null;
                try {
                    photo = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dataUri);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                int dimension = Math.min(photo.getWidth(), photo.getHeight());
                photo = ThumbnailUtils.extractThumbnail(photo, dimension, dimension);
                imgFruit.setImageBitmap(photo);
                photo = Bitmap.createScaledBitmap(photo, ImageSize, ImageSize, false);
                classifyImg(photo);

            }
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

    public void classifyImg(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);

            // ByteBuffer
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * ImageSize * ImageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Get Image Size from device
            // ImageSize is the same as the size for the data model
            int[] intValues = new int[ImageSize * ImageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for (int i = 0; i < ImageSize; i++) {
                for (int j = 0; j <ImageSize; j++) {
                    int val = intValues[pixel++];

                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;

            for (int i = 0; i < confidences.length ; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"Apple", "Banana", "Orange"};
            txtClass.setText(classes[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }
}
