// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include<vector>
#include <string>


int hVec[2] = { 1, -1 };


// th, mean, std, rle
std::vector<float> plotData;

std::vector<int> getLowVector(std::vector<int> fullVector)
{
	std::vector<int> v;
	int iStop = fullVector.size() / 2;
	for (int i = 0; i < iStop; i++)
	{
		float s = fullVector.at(2 * i) + fullVector.at(2 * i + 1);
		v.push_back(s / 2);
	}

	return v;
}


std::vector<int> getHighVector(std::vector<int> fullVector)
{
	std::vector<int> v;
	int iStop = fullVector.size() / 2;
	for (int i = 0; i < iStop; i++)
	{
		float s = fullVector.at(2 * i) - fullVector.at(2 * i + 1);
		v.push_back(s / 2);
	}

	return v;
}

std::vector<int> getLow_USample(std::vector<int> vector_low)
{
	std::vector<int> low_usample;
	int stop = 2 * vector_low.size();
	for (int k = 0; k < stop; k++)
	{
		low_usample.push_back(vector_low.at(k / 2));
	}
	return low_usample;
}

std::vector<int> getHigh_USample(std::vector<int> vector_high)
{
	std::vector<int> high_usample;
	int stop = 2 * vector_high.size();
	for (int k = 0; k < stop; k++)
	{
		high_usample.push_back(vector_high.at(k / 2) * hVec[k % 2]);
	}
	return high_usample;
}


Mat_<int> coef_to_0(Mat_<int> img, int th) {
	Mat_<int> dst(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++) {
			if (abs(img(i, j)) < th) {
				dst(i, j) = 0;

			}
			else {
				dst(i, j) = img(i, j);
			}
		}
	}
	return dst;
}

Mat_<uchar> add128(Mat_<int> src) {
	Mat_<uchar> dst(src.rows, src.cols);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.rows; j++) {
			dst(i, j) = (src(i, j) + 128) % 255;
		}
	}
	return dst;
}


int treshhold(int th, int step)
{
	return th * exp(-step);
}

Mat_<uchar> modifyContrast(Mat_<uchar> img) {
	int imin = 0xffffff, imax = -0xffffff, outMax = 250, outMin = 10;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			//g in max/g in min
			if (img(i, j) < imin) {
				imin = img(i, j);
			}
			if (img(i, j) > imax) {
				imax = img(i, j);
			}
		}
	}
	float decision = (float)(outMax - outMin) / (float)(imax - imin);
	Mat_<uchar>contrast(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			contrast(i, j) = outMin + (img(i, j) - imin) * decision;
		}
	}
	return contrast;

}


void window_imshow_int(std::string name, Mat_<int> img) {
	namedWindow(name, WINDOW_NORMAL);
	imshow(name, add128(img));
}

void window_imshow_uchar(std::string name, Mat_<uchar> img) {
	namedWindow(name, WINDOW_NORMAL);
	imshow(name, img);
}

void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	window_imshow_uchar(name, imgHist);
}

int rowsReturn(int size)
{
	int rows;
	if (size == 1)
	{
		rows = 128;
	}
	if (size == 2)
	{
		rows = 64;
	}
	if (size == 3)
	{
		rows = 32;
	}
	if (size == 4)
	{
		rows = 16;
	}
	if (size == 5)
	{
		rows = 8;
	}
	if (size == 6)
	{
		rows = 4;
	}
	if (size == 7)
	{
		rows = 2;
	}
	if (size == 8)
	{
		rows = 1;
	}
	return rows;
}



// res -> (original - reconstruction)
Mat_<uchar> computeDifference(Mat_<uchar> original, Mat_<uchar> reconstruction)
{
	int rows = original.rows;
	int cols = original.cols;
	Mat_<uchar> res = Mat_<uchar>(rows, cols);
	for (int i = 0; i < original.rows; i++) {
		for (int j = 0; j < original.cols; j++) {
			int diff = abs((original(i, j) - reconstruction(i, j)));
			res(i, j) = (uchar)diff;
		}
	}
	return res;
}

float mean(Mat src) {
	float sum = 0;
	int count = src.rows * src.cols;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			sum += src.at<int>(i, j);
		}
	}

	return sum / count;
}


float std_dev(Mat src) {
	float std = 0;
	float avg = mean(src);
	int count = src.rows * src.cols;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			std += src.at<int>(i, j) * src.at<int>(i, j) - avg * avg;
		}
	}
	std /= count;
	return sqrt(std);
}


void showDifference(Mat_<int> diff) {
	int rows = diff.rows;
	int cols = diff.cols;
	Mat_<uchar> res = Mat_<uchar>(rows, cols);

	for (int i = 0; i < diff.rows; i++) {
		for (int j = 0; j < diff.cols; j++) {
			res(i, j) = diff(i, j) * 10 + 128;
		}
	}
	imshow("Difference", res);
}




int rle_encoding(Mat img) {
	std::vector<int> dst;
	uchar last = img.at<uchar>(0, 0);
	int count = 0;
	dst.push_back((int)last);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			uchar current = img.at<uchar>(i, j);
			if (current != last) {
				dst.push_back(count);
				count = 0;
				dst.push_back((int)current);
				last = current;
				count++;
			}
			else {
				count++;
			}
		}
	}
	return dst.size();
}



std::vector<Mat_<int>> divideIntoFour(Mat_<int> originalImage)
{
	std::vector<Mat_<int>> results;
	int rows = originalImage.rows;
	int cols = originalImage.cols;

	Mat_<float> l = Mat_<int>(rows / 2, cols);
	Mat_<float> h = Mat_<int>(rows / 2, cols);

	for (int c = 0; c < cols; c++)
	{
		std::vector<int> half1Low;
		std::vector<int> half1High;
		for (int r = 0; r < rows; r++)
		{
			half1Low.push_back(originalImage(r, c));
			half1High.push_back(originalImage(r, c));
		}

		half1Low = getLowVector(half1Low);

		half1High = getHighVector(half1High);

		for (int r = 0; r < half1Low.size(); r++)
		{
			l(r, c) = half1Low.at(r);
			h(r, c) = half1High.at(r);
		}
	}

	Mat_<float> ll = Mat_<int>(rows / 2, cols / 2, 255);
	Mat_<float> lh = Mat_<int>(rows / 2, cols / 2, 255);
	Mat_<float> hl = Mat_<int>(rows / 2, cols / 2, 255);
	Mat_<float> hh = Mat_<int>(rows / 2, cols / 2, 255);

	for (int r = 0; r < rows / 2; r++)
	{
		std::vector<int> half1Low;
		std::vector<int> half1High;
		for (int c = 0; c < cols; c++)
		{
			half1Low.push_back(l(r, c));
			half1High.push_back(l(r, c));
		}

		half1Low = getLowVector(half1Low);

		half1High = getHighVector(half1High);


		for (int c = 0; c < half1Low.size(); c++)
		{
			ll(r, c) = half1Low.at(c);
			lh(r, c) = half1High.at(c);
		}
	}

	for (int r = 0; r < rows / 2; r++)
	{
		std::vector<int> half1Low;
		std::vector<int> half1High;
		for (int c = 0; c < cols; c++)
		{
			half1Low.push_back(h(r, c));
			half1High.push_back(h(r, c));
		}

		half1Low = getLowVector(half1Low);

		half1High = getHighVector(half1High);

		for (int c = 0; c < half1Low.size(); c++)
		{
			hl(r, c) = half1Low.at(c);
			hh(r, c) = half1High.at(c);
		}
	}

	results.push_back(ll);
	results.push_back(lh);
	results.push_back(hl);
	results.push_back(hh);

	return results;
}


std::vector<Mat_<int>> divideIntoFourwithTh(Mat_<int> originalImage, int th)
{
	std::vector<Mat_<int>> results;
	std::vector<Mat_<int>> tmp = divideIntoFour(originalImage);

	results.push_back(tmp.at(0));
	results.push_back(coef_to_0(tmp.at(1), th));
	results.push_back(coef_to_0(tmp.at(2), th));
	results.push_back(coef_to_0(tmp.at(3), th));

	return results;
}


Mat_<int> reconstructImage(Mat_<int> ll, Mat_<int> lh, Mat_<int> hl, Mat_<int> hh)
{
	int rows = ll.rows;
	int cols = ll.cols;
	Mat_<int> l = Mat_<int>(rows, cols * 2);
	Mat_<int> h = Mat_<int>(rows, cols * 2);
	for (int r = 0; r < rows; r++)
	{
		std::vector<int> vector_low;
		std::vector<int> vector_high;
		for (int c = 0; c < cols; c++)
		{
			vector_low.push_back(ll(r, c));
			vector_high.push_back(lh(r, c));
		}

		std::vector<int> low_usample = getLow_USample(vector_low);
		std::vector<int> high_usample = getHigh_USample(vector_high);
		for (int c = 0; c < low_usample.size(); c++)
		{
			l(r, c) = low_usample.at(c) + high_usample.at(c);
		}
	}

	for (int r = 0; r < rows; r++)
	{
		std::vector<int> vector_low;
		std::vector<int> vector_high;
		for (int c = 0; c < cols; c++)
		{
			vector_low.push_back(hl(r, c));
			vector_high.push_back(hh(r, c));
		}

		std::vector<int> low_usample = getLow_USample(vector_low);
		std::vector<int> high_usample = getHigh_USample(vector_high);

		for (int c = 0; c < low_usample.size(); c++)
		{
			h(r, c) = low_usample.at(c) + high_usample.at(c);
		}
	}

	Mat_<uchar> reconstructedImage = Mat_<int>(2 * rows, 2 * cols);
	for (int c = 0; c < 2 * cols; c++)
	{
		std::vector<int> vector_low;
		std::vector<int> vector_high;
		for (int r = 0; r < rows; r++)
		{
			vector_low.push_back(l(r, c));
			vector_high.push_back(h(r, c));
		}

		std::vector<int> low_usample = getLow_USample(vector_low);
		std::vector<int> high_usample = getHigh_USample(vector_high);

		for (int r = 0; r < low_usample.size(); r++)
		{
			reconstructedImage(r, c) = low_usample.at(r) + high_usample.at(r);
		}
	}

	return reconstructedImage;
}


Mat_<uchar> combineImage(Mat_<uchar> ll, Mat_<uchar> lh, Mat_<uchar> hl, Mat_<uchar> hh) {

	Mat_<uchar> result(ll.rows * 2, ll.cols * 2);
	ll.copyTo(result(Rect(0, 0, ll.cols, ll.rows)));
	lh.copyTo(result(Rect(lh.rows, 0, lh.cols, lh.rows)));
	hl.copyTo(result(Rect(0, hl.rows, hl.cols, hl.rows)));
	hh.copyTo(result(Rect(hh.rows, hh.cols, hh.cols, hh.rows)));

	return result;
}

void display4Levels() {
	char fname[MAX_PATH];
	int th = 10;
	int size = 4;
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		std::vector<Mat_<int>> finalImg;
		imshow("Original", src);
		for (int i = 0; i < size; i++) {
			th = treshhold(th, i);
			std::vector<Mat_<int>> results = divideIntoFourwithTh(src, th);
			std::cout << th << std::endl;

			Mat_<int> ll = results.at(0);
			Mat_<int> lh = results.at(1);
			Mat_<int> hl = results.at(2);
			Mat_<int> hh = results.at(3);

			Mat_<uchar> pll = ll;
			Mat_<uchar> plh = lh;
			Mat_<uchar> phl = hl;
			Mat_<uchar> phh = hh;

			finalImg.push_back(pll);
			finalImg.push_back(plh);
			finalImg.push_back(phl);
			finalImg.push_back(phh);

			src = pll;
		}
		Mat_<uchar> combineImg;
		Mat_<uchar> reconstructed;
		for (int i = finalImg.size() - 1; i >= 0; i -= 4) {
			Mat_<uchar> pll = finalImg.at(i - 3);
			Mat_<uchar> plh = finalImg.at(i - 2);
			Mat_<uchar> phl = finalImg.at(i - 1);
			Mat_<uchar> phh = finalImg.at(i);

			int rows = rowsReturn(size);
			if (pll.rows == rows) {

				combineImg = combineImage(pll, modifyContrast(plh), modifyContrast(phl), modifyContrast(phh));
			}
			else {
				combineImg = combineImage(combineImg, modifyContrast(plh), modifyContrast(phl), modifyContrast(phh));

			}
		}

		window_imshow_uchar("256x256", combineImg);
		for (int i = finalImg.size() - 1; i >= 0; i -= 4) {
			Mat_<uchar> pll = finalImg.at(i - 3);
			Mat_<uchar> plh = finalImg.at(i - 2);
			Mat_<uchar> phl = finalImg.at(i - 1);
			Mat_<uchar> phh = finalImg.at(i);
			reconstructed = reconstructImage(pll, plh, phl, phh);
			finalImg.at(i - 3) = reconstructed;
		}
		imshow("rec", reconstructed);
		waitKey(0);
	}
}


void showPyramid(std::vector<Mat_<int>> pyramid) {
	int count = 0;

	Mat_<int> ll = pyramid.back();
	pyramid.pop_back();
	window_imshow_uchar("LL", ll);

	while (!pyramid.empty()) {
		Mat_<int> img = pyramid.back();
		pyramid.pop_back();

		std::string s = std::to_string(count);
		window_imshow_uchar(s, add128(img));
		count++;
	}
}


std::vector<Mat_<int>> recursiveDecomposition(Mat_<int> orig)
{
	std::vector<Mat_<int>> result;
	Mat_<uchar> ll = orig.clone();

	while (ll.rows > 2) {
		std::vector<Mat_<int>> divFour = divideIntoFourwithTh(ll, 1);
		ll = divFour.at(0).clone();
		result.push_back(divFour.at(1));
		result.push_back(divFour.at(2));
		result.push_back(divFour.at(3));
		if (ll.rows == 2) {
			result.push_back(ll);
		}
	}

	return result;
}


std::vector<Mat_<int>> recursiveDecompositionNStepsWithTh(Mat_<int> orig, int steps, int initialTh)
{
	std::vector<Mat_<int>> result;
	Mat_<uchar> ll = orig.clone();

	int currentStep = 0;
	while (currentStep < steps) {
		int th = treshhold(initialTh, currentStep);
		std::cout << th << " ";
		std::vector<Mat_<int>> divFour = divideIntoFourwithTh(ll, th);
		ll = divFour.at(0).clone();
		result.push_back(divFour.at(1));
		result.push_back(divFour.at(2));
		result.push_back(divFour.at(3));
		currentStep++;
		if (currentStep == steps) {
			result.push_back(ll);
		}
	}

	std::cout << std::endl;

	return result;
}

// allDecompositions = [LH_128x128, HL_128x128, HH_128x128, LH_64x64, HL_64X64, HH_64X64, ...., LH_2X2, HL_2X2, HH_2X2, LL2x2] -> LL2x2 sa fie ultimul
// LL_4x4 = reconstructImage(LL_2x2, LH_2x2, HL_2x2, HH_2x2)
// ...
// LL_2^n = reconstructImage(LL_2^(n - 1), LH_2^(n - 1), HL_2^(n - 1), HH_2^(n - 1))
Mat_<int> recursiveReconstruction(std::vector<Mat_<int>> allDecompositions)
{

	int sz = allDecompositions.size();
	Mat_<int> ll = reconstructImage(allDecompositions.at(sz - 1), allDecompositions.at(sz - 4), allDecompositions.at(sz - 3), allDecompositions.at(sz - 2));

	for (int i = sz - 5; i >= 0; i -= 3) {
		ll = reconstructImage(ll, allDecompositions.at(i - 2), allDecompositions.at(i - 1), allDecompositions.at(i));
	}

	return ll;
}

void testNLevels() {
	char fname[MAX_PATH];
	int initialTh = 5;
	int levels = 4;
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		std::vector<Mat_<int>> pyramid = recursiveDecompositionNStepsWithTh(src, levels, initialTh);
		showPyramid(pyramid);
		Mat_<int> finalImg = recursiveReconstruction(pyramid);
		Mat_<int> difference = computeDifference(src, finalImg);
		Mat_<uchar> reconstructed = finalImg;

		imshow("Original", src);
		imshow("Reconstruction", reconstructed);
		showDifference(difference);

		waitKey(0);
	}
}


void testNLevelsMetrics() {
	char fname[MAX_PATH];
	std::vector<int> initialTh = {21, 18, 15, 12, 9, 6, 3, 0};
	std::vector<float> stds;
	std::vector<float> avgs;
	std::vector<int> rles;

	int levels = 4;
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("SRC", src);

		int rle = rle_encoding(src);
		std::cout << "Src RLE: " << rle << std::endl;

		for (int i = 0; i < initialTh.size(); i++) {
			std::vector<Mat_<int>> pyramid = recursiveDecompositionNStepsWithTh(src, levels, initialTh[i]);
			Mat_<int> finalImg = recursiveReconstruction(pyramid);
			Mat_<int> difference = computeDifference(src, finalImg);

			float avg = mean(difference);
			avgs.push_back(avg);

			float std = std_dev(difference);
			stds.push_back(std);

			Mat_<uchar> reconstructed = finalImg;

			int rleRec = rle_encoding(reconstructed);
			rles.push_back(rleRec);

			std::string recTitle = "Rec" + std::to_string(initialTh[i]);
			imshow(recTitle, reconstructed);
		}


		for (int i = 0; i < initialTh.size(); i++) {
			std::cout << "TH: " << initialTh[i] << std::endl;
			std::cout << "AVG: " << avgs[i] << std::endl;
			std::cout << "STD: " << stds[i] << std::endl;
			std::cout << "RLE: " << rles[i] << std::endl;
		}


		waitKey(0);
	}
}


void testRecursiveReconstruction() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		std::vector<Mat_<int>> decompositions = recursiveDecomposition(src);
		Mat_<int> finalImg = recursiveReconstruction(decompositions);
		Mat_<uchar> reconstructed = finalImg;

		imshow("Original", src);
		imshow("Reconstruction", reconstructed);
		waitKey(0);
	}
}


void testDecomposition()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		std::vector<Mat_<int>> results = divideIntoFourwithTh(src, 1);
		imshow("Original", src);

		Mat_<int> ll = results.at(0);
		Mat_<int> lh = results.at(1);
		Mat_<int> hl = results.at(2);
		Mat_<int> hh = results.at(3);
		Mat_<uchar> reconstructed = reconstructImage(ll, lh, hl, hh);

		Mat_<uchar> pll = ll;
		Mat_<uchar> plh = lh;
		Mat_<uchar> phl = hl;
		Mat_<uchar> phh = hh;

		window_imshow_uchar("Reconstruction", reconstructed);
		window_imshow_uchar("LL", pll);
		window_imshow_uchar("LH", add128(plh));
		window_imshow_uchar("HL", add128(phl));
		window_imshow_uchar("HH", add128(phh));

		waitKey(0);
	}
}



// se afiseaza src (original), imaginea reconstruita (dupa divizare recursiva si reconstructie recursiva)
//		si imaginea diferenta returnata de functia de mai sus
void testOriginalComparisonWithRes()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		std::vector<Mat_<int>> decompositions = recursiveDecomposition(img);
		Mat_<int> finalImg = recursiveReconstruction(decompositions);
		Mat_<uchar> reconstructed = finalImg;
		Mat_<uchar> dif = computeDifference(img, reconstructed);

		imshow("Original", img);
		imshow("Reconstruction", reconstructed);
		imshow("Difference", dif);

		waitKey(0);
	}
}



void rle_encode_w() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	rle_encoding(img);
	imshow("Src", img);
	waitKey(0);
}




// Se completeaza meniul cu fiecare noua functionalitate
int main()
{
	int op;

	do {
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Decomposition & Reconstruction \n");
		printf(" 2 - Recursive Reconstruction\n");
		printf(" 3 - Recursive N Levels Decomposition \n");
		printf(" 4 - Compare original to reconstructed\n");
		printf(" 5 - Test RLE\n");
		printf(" 6 - Metric\n");
		printf(" 7 - Combine image\n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);

		switch (op) {
		case 1:
			testDecomposition();
			break;
		case 2:
			testRecursiveReconstruction();
			break;
		case 3:
			testNLevels();
			break;
		case 4:
			testOriginalComparisonWithRes();
			break;
		case 5:
			rle_encode_w();
			break;
		case 6:
			testNLevelsMetrics();
			break;
		case 7:
			display4Levels();
			break;
		}
		
	} while (op != 0);

	getchar();

	return 0;
}