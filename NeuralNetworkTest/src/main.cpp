
#include "stdafx.hpp"
#include "mnist/mnist_reader_less.hpp"

using namespace std;
using namespace Eigen;

typedef unsigned char Pixel;

int main(int argc, char* argv[])
{
	//seed random
	srand((unsigned int)time(0));


	//load mnist data
	mnist::filePath = File::getContentDir() + "mnist/";
	mnist::MNIST_dataset<Pixel, Pixel> trainingDataBase = mnist::read_dataset();

	auto b = Eigen::Matrix<Pixel, Constants::imageSize, 1>(trainingDataBase.test_images[0].data());

	std::cout << "Nbr of training images = " << trainingDataBase.training_images.size() << std::endl;
	std::cout << "Nbr of training labels = " << trainingDataBase.training_labels.size() << std::endl;
	std::cout << "Nbr of test images = " << trainingDataBase.test_images.size() << std::endl;
	std::cout << "Nbr of test labels = " << trainingDataBase.test_labels.size() << std::endl;

	//auto t = Map< Matrix<float, Constants::imageSize, 1> >(trainingDataBase.test_images[0].data(), Constants::imageSize);


	VectorXf greyImage(Constants::imageSize);

	File::loadPngAsGrayscale("test.png", &greyImage);


	MatrixXf m = MatrixXf::Random(Constants::imageSize, Constants::imageSize);

	cout << endl << m;
	cout << endl;
	cout << endl << greyImage;
	cout << endl;
	cout << endl << m * greyImage;

	int i;
	cin >> i;
	return 0;
}

