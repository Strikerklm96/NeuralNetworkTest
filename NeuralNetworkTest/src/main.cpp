
#include "Constants.hpp"
#include "Data.hpp"



using namespace std;
using namespace Eigen;

const float e = 2.718281828f;

class Network
{
public:

	std::vector<VectorXf> biases;//biases modify output value for all connections
	std::vector<MatrixXf > weights;//weights modify value for each connection
	MatrixXf answerMatrix;

	Network()
	{
		init(biases, weights);
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


		//answerMatrix = MatrixXf::Zero(numOutputs, numOutputs);
		//for(int i = 0; i < numOutputs; ++i)
		//	answerMatrix(i, i) = 1;

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

			//a = Mat(3,2)
			//b = Vec(2)
			//c = a * b
			weights[layer] = MatrixXf(nextLayerSize, layerSize);
			weights[layer].setRandom();
		}
	}

	float sigmoid(float z)
	{
		return(1.f / (1.0f + std::powf(e, -z)));
	}

	float sigmoidPrime(float z)
	{
		return sigmoid(z) * (1 - sigmoid(z));
	}

	void feedForwardFast(VectorXf* inputLayerValues, const std::vector<VectorXf>& biases, const std::vector<MatrixXf >& weights)
	{
		for(unsigned int i = 0; i < biases.size(); ++i)
		{
			auto bias = biases[i];
			auto weight = weights[i];

			(*inputLayerValues) = weight * (*inputLayerValues) + bias;//still need to add biases, and add sigmoid
		}
	}
	/// <summary>
	/// Train the network.
	/// </summary>
	/// <param name="trainData"></param>
	/// <param name="numEpochs">How many times to go over all the data.</param>
	/// <param name="batchSize">How many attempts to average to find a single delta.</param>
	/// <param name="learningRate"></param>
	/// <param name="testData"></param>
	void train(const DataType& trainData, const int numEpochs, const int samplesPerBatch, const float learningRate, const DataType* testData)
	{
		const DataType& allData = trainData;
		const int numBatches = allData.size() / samplesPerBatch;

		for(int epoch = 0; epoch < numEpochs; ++epoch)
		{
			//group data into batches
			List<DataType> allBatches;
			for(int batch = 0; batch < numBatches; ++batch)
			{
				allBatches.push_back(DataType());
				for(int sampleIndex = 0; sampleIndex < samplesPerBatch; ++sampleIndex)
				{
					int batchOffset = batch * samplesPerBatch;//skip past the samples that we have already done.
					auto& sample = allData[batchOffset + sampleIndex];
					allBatches[batch].push_back(sample);
				}
			}

			//learn
			for(int batch = 0; batch < allBatches.size(); ++batch)
			{
				updateMiniBatch(allBatches[batch], learningRate);
			}

			if(testData != nullptr)
			{

			}

			cout << "\nCompleted Epoch " << epoch;
		}
	}

	void updateMiniBatch(const DataType& batch, float learningRate)
	{

		//set namblas to zero
		auto nambla_b = biases;
		auto nambla_w = weights;
		for(int i = 0; i < nambla_b.size(); ++i)
		{
			nambla_b[i].setZero();
		}
		for(int i = 0; i < nambla_w.size(); ++i)
		{
			nambla_w[i].setZero();
		}

		//for each sample in minibatch
		for(int sample = 0; sample < batch.size(); ++sample)
		{
			auto deltaNambla = backprop(batch[sample].first, batch[sample].second);
			auto& deltaBiases = deltaNambla.first;
			auto& deltaWeights = deltaNambla.second;

			for(int layer = 0; layer < deltaBiases.size(); ++layer)
				nambla_b[layer] += deltaBiases[layer];//add to the total changes in weights and biases

			for(int layer = 0; layer < deltaWeights.size(); ++layer)
				nambla_w[layer] += deltaWeights[layer];

		}

		float learnRateModifier = learningRate / batch.size();

		for(int layer = 0; layer < nambla_b.size(); ++layer)
		{
			biases[layer] -= learnRateModifier * nambla_b[layer];
		}
		for(int layer = 0; layer < weights.size(); ++layer)
		{
			weights[layer] -= learnRateModifier * nambla_w[layer];
		}
	}
	/// <summary>
	/// Returns nambla_b and nambla_w
	/// </summary>
	Pair<List<VectorXf>, List<MatrixXf> > backprop(const VectorXf& guess, int answer)
	{
		return Pair<List<VectorXf>, List<MatrixXf> >();
	}

	float evaluate(List<VectorXf> guesses, DataType answers)
	{
		int totalAttempts = guesses.size();
		int correctGuesses = 0;
		for(int i = 0; i < guesses.size(); ++i)
		{
			AnswerType guess = guesses[i].col(0).maxCoeff();

			if(guess == answers[i].second)
			{
				++correctGuesses;
			}
		}

		return correctGuesses / static_cast<float>(totalAttempts);
	}

	VectorXf costDerivative(const VectorXf& guess, const VectorXf& answer)
	{
		return (guess - answer);//TODO: this is different in the notes! WTF. Ill assume the code is correct and the notes aren't
	}
};

int main(int argc, char* argv[])
{
	//	VectorXf greyImage = Constants::ImageVector();

	//int sln;
	//Data::loadImage("test", &greyImage, &sln);

	srand((unsigned int)time(0));

	Data data;
	if(data.loadData() == false)
		return 1;

	Network n;
	n.train(data.getTrainData(), 30, 10, 3.f, nullptr);


	//data.getTrainImage(45, &greyImage, &sln);

	//n.feedForward(&greyImage, n.biases, n.weights);


	//cout << endl;
	//cout << endl << greyImage;
	//cout << endl;

	int i;
	cin >> i;
	return 0;
}

