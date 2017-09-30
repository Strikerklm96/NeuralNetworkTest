#include "Data.hpp"
#include "LodePng.hpp"
#include "mnist/mnist_reader_less.hpp"
#include <assert.h>
#include "stdafx.hpp"

const std::string dir = "../content/";

const std::string mnist::filePath = dir + "mnist/";


Data::Data()
{
	convertedTest = nullptr;
	convertedTrain = nullptr;
}
const std::string& Data::getContentDir()
{
	return dir;
}
void Data::loadImage(std::string filename, ActiveType* greyImage, int* solution)
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
	ActiveType& t = *greyImage;
	for(unsigned int i = 0; i < rgbaImage.size(); i += 4)
	{
		t(i/4,0) = 1 - (((rgbaImage[i] + rgbaImage[i + 1] + rgbaImage[i + 2]) / 3.f) / 255.f);
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

void Data::getTrainImage(int index, ActiveType* greyImage, int* solution, float rot) const
{
	getStuff(index, greyImage, dataBase.training_images[index], rot);
	*solution = dataBase.training_labels[index];
}
void Data::getTestImage(int index, ActiveType* greyImage, int* solution, float rot) const
{
	getStuff(index, greyImage, dataBase.test_images[index], rot);
	*solution = dataBase.test_labels[index];
}
void Data::getStuff(int index, ActiveType* greyImage, const std::vector<unsigned char>& image, float rot) const
{
	for(unsigned i = 0; i < image.size(); ++i)
	{
		(*greyImage)(i,0) = image[i] / 255.f;
	}
}
DataType& Data::getTestData()
{
	if(convertedTest == nullptr)
	{
		convertedTest = new DataType();

		const int numTestImages = dataBase.test_images.size();
		for(int i = 0; i < numTestImages; ++i)
		{
			ActiveType image(Constants::imageSize,1);
			AnswerType answer;
			float rot = 0;
			getTestImage(i, &image, &answer, rot);

			(*convertedTest).push_back(Pair<ActiveType, AnswerType>(image, answer));
		}
	}

	return *convertedTest;
}
DataType& Data::getTrainData()
{
	if(convertedTrain == nullptr)
	{
		convertedTrain = new DataType();

		const int numTrainImages = dataBase.training_images.size();
		for(int i = 0; i < numTrainImages; ++i)
		{
			ActiveType image(Constants::imageSize, 1);
			AnswerType answer;
			float rot = 0;
			getTrainImage(i, &image, &answer, rot);

			(*convertedTrain).push_back(Pair<ActiveType, AnswerType>(image, answer));
		}
	}

	return *convertedTrain;
}
