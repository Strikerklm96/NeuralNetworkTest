
#include "Constants.hpp"
#include "Data.hpp"



using namespace std;
using namespace Eigen;

int main(int argc, char* argv[])
{
	//seed random
	srand((unsigned int)time(0));

	Data data;
	if(data.loadData() == false)
		return 1;


	//load mnist data

	//auto b = ImageVector(trainingDataBase.test_images[0].data());

	//std::cout << "Nbr of training images = " << trainingDataBase.training_images.size() << std::endl;
	//std::cout << "Nbr of training labels = " << trainingDataBase.training_labels.size() << std::endl;
	//std::cout << "Nbr of test images = " << trainingDataBase.test_images.size() << std::endl;
	//std::cout << "Nbr of test labels = " << trainingDataBase.test_labels.size() << std::endl;

	//auto t = Map< Matrix<float, Constants::imageSize, 1> >(trainingDataBase.test_images[0].data(), Constants::imageSize);


	ImageVector greyImage;

	int sln;
	Data::loadPngAsGrayscale("test", &greyImage, &sln);


	MatrixXf m = MatrixXf::Random(Constants::imageSize, Constants::imageSize);

	cout << endl << m;
	cout << endl;
	cout << endl << greyImage;
	cout << endl;/*
	cout << endl << m * greyImage;*/

	int i;
	cin >> i;
	return 0;
}

