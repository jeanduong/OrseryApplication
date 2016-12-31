package com.example.jeanduong.ocrapplication;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.LinkedList;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static org.opencv.core.Core.bitwise_not;


public class GrayActivity extends AppCompatActivity {

    static final String TAG = "Business OCR";

    TessBaseAPI tess_engine; //Tess API reference
    String datapath = ""; //path to folder containing language data file
    String lang = "fra";

    // To be placed before onCreate!!!
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "Library loaded successfully");
                    Toast.makeText(GrayActivity.this, "OpenCV loaded successfully", Toast.LENGTH_LONG).show();
                    torture();
                } break;
                default:
                {
                    Log.i("OpenCV", "Failed to load");
                    Toast.makeText(GrayActivity.this, "Failed to load OpenCV", Toast.LENGTH_LONG).show();
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gray);
    }

    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private void torture()
    {
        ///////////
        // Setup //
        ///////////

        // Tesseract setup
        tess_engine = new TessBaseAPI();
        datapath = getFilesDir() + "/tesseract/";
        tess_engine.init(datapath, lang);

        checkFile(new File(datapath + "tessdata/")); //make sure training data has been copied

        // Bitmap is only needed to load image from ressources and create data array.
        // It can be deleted after that, since only Mat objects will be used for processing

        //Bitmap img_rgb = BitmapFactory.decodeResource(getResources(), R.drawable.dooblink_crop);
        //Bitmap img_rgb = BitmapFactory.decodeResource(getResources(), R.drawable.dooblink_crop_1000);
        //Bitmap img_rgb = BitmapFactory.decodeResource(getResources(), R.drawable.pulsalys_crop);
        Bitmap img_rgb = BitmapFactory.decodeResource(getResources(), R.drawable.pulsalys_crop_1000);
        //Bitmap img_rgb = BitmapFactory.decodeResource(getResources(), R.drawable.mills_crop);

        final int h = img_rgb.getHeight();
        final int w = img_rgb.getWidth();
        Mat mat_rgb = new Mat(h, w, CvType.CV_8UC3);

        Utils.bitmapToMat(img_rgb, mat_rgb);
        img_rgb.recycle(); // May make the app crash
        //Imgproc.GaussianBlur(mat_rgb, mat_rgb, new Size(5, 5), 0.0); // Blur to remove some noise

        ////////////////////////////////
        // Leydier's pseudo-luminance //
        ////////////////////////////////

        Imgproc.cvtColor(mat_rgb, mat_rgb, Imgproc.COLOR_BGR2RGB); // OpenCV color image are BGR!!!
        Mat mat_lprime = MakeLPrime(mat_rgb);
        //Imgproc.GaussianBlur(mat_lprime, mat_lprime, new Size(5, 5), 0.0); // Blur to remove some noise

        //////////////////////////////
        // Run OCR and extract data //
        //////////////////////////////

        Bitmap img_out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
/*
        // OCR over RGB image
        Utils.matToBitmap(mat_rgb, img_out);
        tess_engine.setImage(img_out);
        String txt_color = tess_engine.getUTF8Text();
        Log.i(TAG, "(RGB blurred)\n" +  txt_color);
*/
        mat_rgb.release(); // May make the app crash

        // OCR over grayscaled image
        Utils.matToBitmap(mat_lprime, img_out);
        tess_engine.setImage(img_out);
        String txt_gray = tess_engine.getUTF8Text();
        Log.i(TAG, "(grayscaled)\n\n" + txt_gray);

        ((TextView) findViewById(R.id.content_gray)).setText(txt_gray);
        ((ImageView) findViewById(R.id.gray_display_view_name)).setImageBitmap(img_out);
/*
        // OCR over binary image
        Utils.matToBitmap(MakeBinary(mat_lprime), img_out);
        tess_engine.setImage(img_out);
        String txt_bin = tess_engine.getUTF8Text();
        Log.i(TAG, "(binarized)\n\n" + txt_bin);

        ((TextView) findViewById(R.id.content_binary)).setText(txt_bin);
*/
        tess_engine.end();
    }



    ////////////////
    // ZOI fusion //
    ////////////////

    private boolean Custom_proximity_heuristic(android.graphics.Rect rct_ref, android.graphics.Rect rct_other)
    {
        final double top_ref = rct_ref.top;
        final double bottom_ref = rct_ref.bottom;
        final double top_other = rct_other.top;
        final double bottom_other = rct_other.bottom;

        // Horizontal projections should overlap
        if ((bottom_other < top_ref) || (top_other > bottom_ref)) return false;

        final double height_ref = rct_ref.height();
        final double height_other = rct_other.height();
        final double vertical_range_overlap = min(bottom_ref, bottom_other) - max(top_ref, top_other) + 1;

        // Overlap range should be significant
        if (vertical_range_overlap < min(height_ref, height_other) / 2.0) return false;

        final double left_ref = rct_ref.left;
        final double right_ref = rct_ref.right;
        final double left_other = rct_other.left;
        final double right_other = rct_other.right;
        final double width_ref = rct_ref.width();
        final double width_other = rct_other.width();

        // Gap between candidates should not be greater than heights
        if (left_ref < left_other)
        {
            if (left_other - right_ref > max(height_ref, height_other))
                return false;
        }
        else if (left_ref - right_other > max(height_ref, height_other))
            return false;

        return true;
    }

    private LinkedList<android.graphics.Rect> Gather_intersection(LinkedList<android.graphics.Rect> rectangles)
    {
        LinkedList<android.graphics.Rect> rcts = new LinkedList<android.graphics.Rect>(rectangles);
        LinkedList<android.graphics.Rect> tmp = new LinkedList<android.graphics.Rect>();
        LinkedList<android.graphics.Rect> res = new LinkedList<android.graphics.Rect>();

        while (rcts.size() > 0)
        {
            tmp.clear();
            android.graphics.Rect z = rcts.get(0);

            for (int k = 1; k < rcts.size(); ++k)
            {
                android.graphics.Rect r = rcts.get(k);

                if (Custom_proximity_heuristic(z, r))
                    z.union(r);
                else
                    tmp.add(r);
            }

            res.add(z);
            rcts.clear();
            rcts.addAll(tmp);
        }

        return res;
    }

    private LinkedList<android.graphics.Rect> Gather_intersection_ultimate(LinkedList<android.graphics.Rect> rectangles)
    {
        LinkedList<android.graphics.Rect> rcts = new LinkedList<android.graphics.Rect>(rectangles);
        LinkedList<android.graphics.Rect> res = Gather_intersection(rectangles);

        while (res.size() != rcts.size())
        {
            rcts.clear();
            rcts.addAll(res);
            res.clear();
            res = Gather_intersection(rcts);
        }

        return res;
    }

    private LinkedList<android.graphics.Rect> Gather_alignment(LinkedList<android.graphics.Rect> rectangles)
    {
        LinkedList<android.graphics.Rect> rcts = new LinkedList<android.graphics.Rect>(rectangles);
        LinkedList<android.graphics.Rect> tmp = new LinkedList<android.graphics.Rect>();
        LinkedList<android.graphics.Rect> res = new LinkedList<android.graphics.Rect>();

        while (rcts.size() > 0)
        {
            tmp.clear();
            android.graphics.Rect z = rcts.get(0);

            for (int k = 1; k < rcts.size(); ++k)
            {
                android.graphics.Rect r = rcts.get(k);

                if (Custom_proximity_heuristic(z, r))
                    z.union(r);
                else
                    tmp.add(r);
            }

            res.add(z);
            rcts.clear();
            rcts.addAll(tmp);
        }

        return res;
    }

    private LinkedList<android.graphics.Rect> Gather_heuristic(LinkedList<android.graphics.Rect> rectangles)
    {
        boolean reloop = true;
        LinkedList<android.graphics.Rect> res = new LinkedList<android.graphics.Rect>();
        LinkedList<android.graphics.Rect> tmp = new LinkedList<android.graphics.Rect>();

        res.addAll(rectangles);

        int card_before = 0;
        int card_after = 0;

        do {
            card_before = res.size();
            tmp.clear();
            tmp.addAll(Gather_intersection(res));
            res.clear();
            res.addAll(Gather_alignment(tmp));
            card_after = res.size();

            reloop = (card_after != card_before);
        }
        while (reloop);

        return res;
    }

    /////////////////////////////
    // Custom image processing //
    /////////////////////////////

    private Mat MakeLPrime(Mat mat_rgb)
    {
        final int h = mat_rgb.rows();
        final int w = mat_rgb.cols();

        Mat mat_L_prime = new Mat(h, w, CvType.CV_8UC1);

        double red, green, blue, s, v, mi, ma;
        double[] triplet;

        // Time consuming version

        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
            {
                triplet = mat_rgb.get(r, c);
                red = triplet[0];
                green = triplet[1];
                blue = triplet[2];

                ma = max(red, max(green, blue));
                mi = min(red, min(green, blue));

                if (ma > 0.0)
                    s = 255.0 * (1.0 - (mi / ma));
                else
                    s = 0.0;

                v = ((255.0 - s) * (red * 0.299 + green * 0.587 + blue * 0.114)) / 255.0;

                mat_L_prime.put(r, c, (int)min(255.0, max(0.0, v)));
            }

        return mat_L_prime;
    }

    private Mat MakeBinary(Mat mat_gray)
    {
        final int h = mat_gray.rows();
        final int w = mat_gray.cols();

        Mat mat_bin = new Mat(h, w, CvType.CV_8UC1, new Scalar(255));

        ////////////////////
        // MSER detection //
        ////////////////////

        MatOfKeyPoint mat_mser_keypoints = new MatOfKeyPoint();
        FeatureDetector detector = FeatureDetector.create(FeatureDetector.MSER);

        detector.detect(mat_gray, mat_mser_keypoints);

        Log.i(TAG, "MSER ZOIs : " + mat_mser_keypoints.rows());

        KeyPoint[] array_mser_keypoints = mat_mser_keypoints.toArray();
        LinkedList<Rect> mser_zois = new LinkedList<Rect>();
        MatOfDouble mser_radii_sample = new MatOfDouble(array_mser_keypoints.length, 1);
        double[] mser_radii_array = new double[array_mser_keypoints.length];

        for (int k = 0; k < array_mser_keypoints.length; ++k)
        {
            KeyPoint kp = array_mser_keypoints[k];
            int half_size = (int)(kp.size / 2.0);
            Point pt = kp.pt;
            int x = (int) pt.x;
            int y = (int) pt.y;

            int l = (int) max(0, x - half_size);
            int r = (int) min(w - 1, x + half_size);
            int t = (int) max(0, y - half_size);
            int b = (int) min(h - 1, y + half_size);

            mser_radii_sample.put(k, 0, kp.size);
            mser_radii_array[k] = kp.size;
            mser_zois.add(new Rect(l, t, r, b));
        }

        // Gathering rectangles

        mser_zois = Gather_heuristic(mser_zois);

        Log.i(TAG, "MSER ZOIs : " + mser_zois.size() + " (after fusion)");

        for (int k = 0; k < mser_zois.size(); ++k)
        {
            Rect zoi = mser_zois.get(k);
            int x = zoi.left;
            int y = zoi.top;
            int z_w = zoi.width();
            int z_h = zoi.height();

            Mat mat_zoi = new Mat(z_h, z_w, CvType.CV_8UC1, new Scalar(255));

            int rr = y;
            int cc = x;

            for (int r = 0; r < z_h; ++r)
            {
                for (int c = 0; c < z_w; ++c) {
                    mat_zoi.put(r, c, mat_gray.get(rr, cc));
                    ++cc;
                }

                ++rr;
                cc = x;
            }

            Imgproc.threshold(mat_zoi, mat_zoi, 0.0, 255.0, Imgproc.THRESH_OTSU);

            rr = y;
            cc = x;

            for (int r = 0; r < z_h; ++r)
            {
                for (int c = 0; c < z_w; ++c)
                {
                    if (mat_zoi.get(r, c)[0] == 0.0)
                        mat_bin.put(rr, cc, 0);
                    else
                        mat_bin.put(rr, cc, 255);

                    ++cc;
                }

                ++rr;
                cc = x;
            }
        }

        return mat_bin;
    }

    //////////////////////////////////////
    // Data handling for Tesseract user //
    //////////////////////////////////////

    private void copyFiles() {
        try {
            //location we want the file to be at
            String filepath = datapath + "/tessdata/" + lang + ".traineddata";

            //get access to AssetManager
            AssetManager assetManager = getAssets();

            //open byte streams for reading/writing
            InputStream instream = assetManager.open("tessdata/" + lang + ".traineddata");
            OutputStream outstream = new FileOutputStream(filepath);

            //copy the file to the location specified by filepath
            byte[] buffer = new byte[1024];
            int read;
            while ((read = instream.read(buffer)) != -1) {
                outstream.write(buffer, 0, read);
            }
            outstream.flush();
            outstream.close();
            instream.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void checkFile(File dir) {
        //directory does not exist, but we can successfully create it
        if (!dir.exists()&& dir.mkdirs()){
            copyFiles();
        }
        //The directory exists, but there is no data file in it
        if(dir.exists()) {
            String datafilepath = datapath+ "/tessdata/" + lang + ".traineddata";
            File datafile = new File(datafilepath);
            if (!datafile.exists()) {
                copyFiles();
            }
        }
    }
}
