
#include "Constants.hpp"
#include "Data.hpp"



using namespace std;
using namespace Eigen;

const float e = 2.718281828f;

float sigmoid(float z)
{
	return(1.f / (1.0f + std::powf(e, -z)));
}


int main(int argc, char* argv[])
{
	const int inputLayerSize = 2;// Constants::imageSize;
	const int hiddenLayerSize = 5;// 30;
	const int numOutputs = 1;// 10;

	std::vector<int> sizes;
	sizes.push_back(inputLayerSize);
	sizes.push_back(hiddenLayerSize);
	sizes.push_back(numOutputs);

	int numLayers = sizes.size();

	std::vector<VectorXf> biases;//biases modify output value for all connections

	int start = 1;//first layer is input layer, so it doesnt modify its output (the output is the value of the pixels)
	for(int i = start; i < numLayers; ++i)//for each neuron in the network
	{
		auto biasLayer = VectorXf(sizes[i]);//give it a bias
		biasLayer.setRandom();
		biases.push_back(biasLayer);
	}

	std::vector<std::vector<VectorXf> > weights;//weights modify value for each connection
	weights.resize(numLayers - 1);
	for(int layer = 0; layer < numLayers - 1; layer++)//for each layer
	{
		auto layerSize = sizes[layer];
		auto& nextLayerSize = sizes[layer + 1];

		for(int neuron = 0; neuron < nextLayerSize; neuron++)//get a neuron in the next layer
		{
			auto currentLayerWeights = VectorXf(layerSize);//give it a weight for each neuron in this layer
			currentLayerWeights.setRandom();
			weights[layer].push_back(currentLayerWeights);
		}
	}

	float t = biases[0][0];

	ImageVector greyImage;

	int sln;
	Data::loadImage("test", &greyImage, &sln);
	
	srand((unsigned int)time(0));

	Data data;
	if(data.loadData() == false)
		return 1;

	data.getTrainImage(45, &greyImage, &sln);

	//seed random


	//for(int i = 0; i<
	//std::vector<

	//load mnist data

	//auto b = ImageVector();

	//std::cout << "Nbr of training images = " << trainingDataBase.training_images.size() << std::endl;
	//std::cout << "Nbr of training labels = " << trainingDataBase.training_labels.size() << std::endl;
	//std::cout << "Nbr of test images = " << trainingDataBase.test_images.size() << std::endl;
	//std::cout << "Nbr of test labels = " << trainingDataBase.test_labels.size() << std::endl;
/*
	auto t = Map< Matrix<unsigned char, Constants::imageSize, 1> >(data.dataBase.test_images[0].data(), Constants::imageSize);*/





//	cout << endl << m;
	cout << endl;
	cout << endl << greyImage;
	cout << endl;/*
	cout << endl << m * greyImage;*/

	int i;
	cin >> i;
	return 0;
}

