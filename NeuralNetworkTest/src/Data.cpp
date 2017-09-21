#include "Data.hpp"
#include "LodePng.hpp"
#include "mnist/mnist_reader_less.hpp"
#include <assert.h>
#include "stdafx.hpp"

const std::string dir = "../content/";

const std::string mnist::filePath = dir + "mnist/";


const std::string& Data::getContentDir()
{
	return dir;
}
void Data::loadImage(std::string filename, Eigen::VectorXf* greyImage, int* solution)
{
	std::vector<unsigned char> rgbaImage;
	unsigned width, height;
	unsigned error = lodepng::decode(rgbaImage, width, height, Data::getContentDir() + filename + ".png");

	if(error)
	{
		std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		assert(error == false);
	}


	//image now in RGBA, RGBA etc.

	//convert to greyscale
	for(unsigned int i = 0; i < rgbaImage.size(); i += 4)
	{
		(*greyImage)[i / 4] = ((rgbaImage[i] + rgbaImage[i + 1] + rgbaImage[i + 2]) / 3.f)/255.f;
	}
}


bool Data::loadData()
{
	dataBase = mnist::read_dataset();

	if(dataBase.training_images.size() == 60000)
		return true;
	else
		return false;
}

void Data::getTrainImage(int index, Eigen::VectorXf* greyImage, int* solution)
{
	getStuff(index, greyImage, dataBase.training_images[index]);
	*solution = dataBase.training_labels[index];
}
void Data::getTestImage(int index, Eigen::VectorXf* greyImage, int* solution)
{
	getStuff(index, greyImage, dataBase.test_images[index]);
	*solution = dataBase.test_labels[index];
}
void Data::getStuff(int index, Eigen::VectorXf* greyImage, const std::vector<unsigned char>& image)
{
	for(unsigned i = 0; i < image.size(); ++i)
	{
		(*greyImage)[i] = image[i] / 255.f;
	}
}
