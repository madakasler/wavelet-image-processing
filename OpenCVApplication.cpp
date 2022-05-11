// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include<vector>
#include <string>

int hVec[2] = { 1, -1 };


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

Mat_<uchar> modifyContrast(Mat_<int> img) {
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


void window_imshow_int(std::string name, Mat_<int> img) {
	namedWindow(name, WINDOW_NORMAL);
	imshow(name, add128(img));
}

void window_imshow_uchar(std::string name, Mat_<uchar> img) {
	namedWindow(name, WINDOW_NORMAL);
	imshow(name, img);
}

// res -> (original - reconstruction) * 10 + 128
Mat_<uchar> computeDifference(Mat_<uchar> original, Mat_<uchar> reconstruction)
{
	int rows = original.rows;
	int cols = original.cols;
	Mat_<uchar> res = Mat_<uchar>(rows, cols);
	for (int i = 0; i < original.rows; i++) {
		for (int j = 0; j < original.cols; j++) {
			res(i, j) = (original(i, j) - reconstruction(i, j)) * 10 + 128;
		}
	}
	//res = res * 10 + 128;
	return res;
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
	Mat_<float> afisLLmatTh = Mat_<int>(rows / 2, cols / 2, 255);
	Mat_<float> afisLHmatTh = Mat_<int>(rows / 2, cols / 2, 255);
	Mat_<float> afisHLmatTh = Mat_<int>(rows / 2, cols / 2, 255);
	Mat_<float> afisHHmatTh = Mat_<int>(rows / 2, cols / 2, 255);



	results.push_back(ll);
	results.push_back(lh);
	results.push_back(hl);
	results.push_back(hh);

	return results;
}


std::vector<Mat_<int>> divideIntoFourwithTh(Mat_<int> originalImage, int th)
{
	std::vector<Mat_<int>> results;
	int rows = originalImage.rows;
	int cols = originalImage.cols;

	Mat_<int> l = Mat_<int>(rows / 2, cols);
	Mat_<int> h = Mat_<int>(rows / 2, cols);

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

	Mat_<int> ll = Mat_<int>(rows / 2, cols / 2, 255);
	Mat_<int> lh = Mat_<int>(rows / 2, cols / 2, 255);
	Mat_<int> hl = Mat_<int>(rows / 2, cols / 2, 255);
	Mat_<int> hh = Mat_<int>(rows / 2, cols / 2, 255);

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
	results.push_back(coef_to_0(lh, th));
	results.push_back(coef_to_0(hl, th));
	results.push_back(coef_to_0(hh, th));

	return results;
}


Mat_<uchar> reconstructImage(Mat_<int> ll, Mat_<int> lh, Mat_<int> hl, Mat_<int> hh)
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

// allDecompositions = [LH_128x128, HL_128x128, HH_128x128, LH_64x64, HL_64X64, HH_64X64, ...., LH_2X2, HL_2X2, HH_2X2, LL2x2] -> LL2x2 sa fie ultimul
// LL_4x4 = reconstructImage(LL_2x2, LH_2x2, HL_2x2, HH_2x2)
// ...
// LL_2^n = reconstructImage(LL_2^(n - 1), LH_2^(n - 1), HL_2^(n - 1), HH_2^(n - 1))
Mat_<uchar> recursiveReconstruction(std::vector<Mat_<int>> allDecompositions)
{

	int sz = allDecompositions.size();
	Mat_<int> ll = reconstructImage(allDecompositions.at(sz - 1), allDecompositions.at(sz - 4), allDecompositions.at(sz - 3), allDecompositions.at(sz - 2));

	for (int i = sz - 5; i >= 0; i -= 3) {
		ll = reconstructImage(ll, allDecompositions.at(i - 2), allDecompositions.at(i - 1), allDecompositions.at(i));
	}

	return ll;
}


void display4Levels() {
	char fname[MAX_PATH];
	int initialTh = 5;
	int levels = 4;
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat_<uchar> scr_copy = src.clone();
		std::vector<Mat_<int>> finalImg;
		imshow("Original", src);
		for (int i = 0; i < levels; i++) {
			int th = treshhold(initialTh, i);
			std::cout << th << std::endl;

			std::vector<Mat_<int>> results = divideIntoFourwithTh(src, th);

			Mat_<int> ll = results.at(0);
			Mat_<int> lh = results.at(1);
			Mat_<int> hl = results.at(2);
			Mat_<int> hh = results.at(3);

			finalImg.push_back(ll);
			finalImg.push_back(lh);
			finalImg.push_back(hl);
			finalImg.push_back(hh);

			src = ll;
		}
		Mat_<uchar> combineImg;
		Mat_<uchar> reconstructed;
		for (int i = finalImg.size() - 1; i >= 0; i -= 4) {
			Mat_<int> pll = finalImg.at(i - 3);
			Mat_<int> plh = finalImg.at(i - 2);
			Mat_<int> phl = finalImg.at(i - 1);
			Mat_<int> phh = finalImg.at(i);

			int rows = rowsReturn(levels);
			if (pll.rows == rows) {
				combineImg = combineImage(pll, modifyContrast(plh), modifyContrast(phl), modifyContrast(phh));
			}
			else {
				combineImg = combineImage(combineImg, modifyContrast(plh), modifyContrast(phl), modifyContrast(phh));
			}
		}

		window_imshow_uchar("all levels", combineImg);

		for (int i = finalImg.size() - 1; i >= 0; i -= 4) {
			Mat_<int> pll = finalImg.at(i - 3);
			Mat_<int> plh = finalImg.at(i - 2);
			Mat_<int> phl = finalImg.at(i - 1);
			Mat_<int> phh = finalImg.at(i);

			reconstructed = reconstructImage(pll, plh, phl, phh);
			finalImg.at(i - 3) = reconstructed;
		}

		Mat_<uchar> difference = computeDifference(scr_copy, reconstructed);
		window_imshow_uchar("diff", difference);
		imshow("rec", reconstructed);
		waitKey(0);
	}
}
// result = [LH_128x128, HL_128x128, HH_128x128, LH_64x64, HL_64X64, HH_64X64, ...., LH_2X2, HL_2X2, HH_2X2, LL_2x2]
// LL_2^n x 2^n -> LL_2^(n - 1), LH_2^(n - 1), HL_2^(n - 1), HH_2^(n - 1)




// result = [LL_128x128,LH_128x128, HL_128x128, HH_128x128,LL_64x64, LH_64x64, HL_64X64, HH_64X64, ...., LL_16x16, LH_16X16, HL_16X16, HH_16X16]



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



void rle_encoding(Mat img) {

	std::vector<int> dst;

	uchar last = img.at<uchar>(0, 0);
	int count = 0;
	dst.push_back((int)last);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			uchar current = img.at<uchar>(i, j);
			if (current != last) {
				std::cout << (int)last << " " << count << std::endl;
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
			display4Levels();
			break;
		case 4:
			testOriginalComparisonWithRes();
			break;
		case 5:
			rle_encode_w();
			break;


		}
	} while (op != 0);

	getchar();

	return 0;
}