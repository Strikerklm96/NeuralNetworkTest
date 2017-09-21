
#include "Constants.hpp"
#include "Data.hpp"



using namespace std;
using namespace Eigen;

const float e = 2.718281828f;

float sigmoid(float z)
{
	return(1.f / (1.0f + std::powf(e, -z)));
}

void init(std::vector<VectorXf>& biases, std::vector<MatrixXf >& weights)
{
	const int inputLayerSize = Constants::imageSize;
	const int hiddenLayerSize = 30;
	const int numOutputs = 10;

	std::vector<int> sizes;
	sizes.push_back(inputLayerSize);
	sizes.push_back(hiddenLayerSize);
	sizes.push_back(numOutputs);

	int numLayers = sizes.size();

	int start = 1;//first layer is input layer, so it doesnt modify its output (the output is the value of the pixels)
	for(int i = start; i < numLayers; ++i)//for each neuron in the network
	{
		auto biasLayer = VectorXf(sizes[i]);//give it a bias
		biasLayer.setRandom();
		biases.push_back(biasLayer);
	}

	weights.resize(numLayers - 1);
	for(int layer = 0; layer < numLayers - 1; layer++)//for each layer
	{
		auto layerSize = sizes[layer];
		auto& nextLayerSize = sizes[layer + 1];

		//a = (3,2)
		//b = (2)
		//c = a * b
		weights[layer] = MatrixXf(nextLayerSize, layerSize);
		weights[layer].setRandom();
	}
}

void feedForward(VectorXf* inputLayerValues, const std::vector<VectorXf>& biases, const std::vector<MatrixXf >& weights)
{
	for(unsigned int i = 0; i < biases.size(); ++i)
	{
		auto bias = biases[i];
		auto weight = weights[i];

		(*inputLayerValues) = weight * (*inputLayerValues) + bias;//still need to add biases, and add sigmoid
	}
}


int main(int argc, char* argv[])
{


	std::vector<VectorXf> biases;//biases modify output value for all connections
	std::vector<MatrixXf > weights;//weights modify value for each connection

	init(biases, weights);

	VectorXf greyImage = Constants::ImageVector();

	int sln;
	Data::loadImage("test", &greyImage, &sln);

	srand((unsigned int)time(0));

	Data data;
	if(data.loadData() == false)
		return 1;

	data.getTrainImage(45, &greyImage, &sln);

	feedForward(&greyImage, biases, weights);


	cout << endl;
	cout << endl << greyImage;
	cout << endl;

	int i;
	cin >> i;
	return 0;
}

