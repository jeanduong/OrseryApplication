package com.example.jeanduong.ocrapplication;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import static java.lang.Math.max;
import static java.lang.Math.min;

public class MainActivity extends Activity {

    Bitmap img_rgb; //Source image
    TessBaseAPI tess_engine; //Tess API reference
    String datapath = ""; //path to folder containing language data file
    String lang = "fra";

    final String TAG = "Business card OCR";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ImageView vw = (ImageView) findViewById(R.id.display_view_name);

        img_rgb = BitmapFactory.decodeResource(getResources(), R.drawable.dooblink_crop);
        //img_rgb = BitmapFactory.decodeResource(getResources(), R.drawable.pulsalys_crop);
        //img_rgb = BitmapFactory.decodeResource(getResources(), R.drawable.mills_crop);

        vw.setImageBitmap(img_rgb);
    }

    public void to_gray(View vw)
    {
        Intent itt = new Intent(this, GrayActivity.class);

        if (itt.resolveActivity(getPackageManager()) != null)
            startActivity(itt);
    }

}
