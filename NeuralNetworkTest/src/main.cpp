
#include "stdafx.hpp"
#include "mnist/mnist_reader_less.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[])
{	

    //// Load MNIST data
    //mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

	mnist::filePath = File::getContentDir() + "mnist/";
	auto dataset = mnist::read_dataset();

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;




	VectorXf greyImage(Constants::imageSize());

	File::loadPngAsGrayscale("test.png", &greyImage);

	srand((unsigned int)time(0));

	MatrixXf m = MatrixXf::Random(Constants::imageSize(), Constants::imageSize());

	cout << endl << m;
	cout << endl;
	cout << endl << greyImage;
	cout << endl;
	cout << endl << m * greyImage;

	int i;
	cin >> i;
	return 0;
}

