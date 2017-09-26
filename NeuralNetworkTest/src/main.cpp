
#include "Constants.hpp"
#include "Data.hpp"



using namespace std;
using namespace Eigen;

const float e = 2.718281828f;


float sigmoid(float z)
{
	return(1.f / (1.0f + std::powf(e, -z)));
}

float sigmoidPrime(float z)
{
	return sigmoid(z) * (1 - sigmoid(z));
}

class Network
{
public:

	const int inputLayerSize = 1;// Constants::imageSize;
	const int hiddenLayerSize = 1;// 30;
	const int numOutputs = 2;// 10;
	std::vector<int> layerSizes;

	BiasType biases;//biases modify output value for all connections
	WeightType weights;//weights modify value for each connection
	MatrixXf answerMatrix;

	const MatrixXf& getAnswerMat() const
	{
		return answerMatrix;
	}

	Network()
	{
		init(biases, weights);
	}
	void init(std::vector<VectorXf>& biases, std::vector<MatrixXf >& weights)
	{

		srand((unsigned int)time(0));

		layerSizes.push_back(inputLayerSize);
		layerSizes.push_back(hiddenLayerSize);
		layerSizes.push_back(numOutputs);


		answerMatrix = MatrixXf::Zero(numOutputs, numOutputs);
		for(unsigned i = 0; i < numOutputs; ++i)
			answerMatrix(i, i) = 1;

		unsigned numLayers = layerSizes.size();

		unsigned start = 1;//first layer is input layer, so it doesnt modify its output (the output is the value of the pixels)
		for(unsigned i = start; i < numLayers; ++i)//for each neuron in the network
		{
			auto biasLayer = VectorXf(layerSizes[i]);//give it a bias
			biasLayer.setRandom();
			biases.push_back(biasLayer);
		}

		weights.resize(numLayers - 1);
		for(unsigned layer = 0; layer < numLayers - 1; layer++)//for each layer
		{
			auto layerSize = layerSizes[layer];
			auto& nextLayerSize = layerSizes[layer + 1];

			//a = Mat(3,2)
			//b = Vec(2)
			//c = a * b
			weights[layer] = MatrixXf(nextLayerSize, layerSize);
			weights[layer].setRandom();
		}
	}


	void feedForward(ActiveType* activation, const BiasType& biases, const WeightType& weights, List<ActiveType>* recordActivations, List<ActiveType>* recordZ)
	{
		if(recordActivations)
			recordActivations->push_back(*activation);

		for(unsigned int i = 0; i < biases.size(); ++i)
		{
			auto bias = biases[i];
			auto weight = weights[i];

			(*activation) = weight * (*activation) + bias;// z (weighted input)

			if(recordZ)
				recordZ->push_back(*activation);

			(*activation) = (*activation).unaryExpr(&sigmoid);// activation

			if(recordActivations)
				recordActivations->push_back(*activation);
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
	void train(const DataType& trainData, const unsigned numEpochs, const unsigned samplesPerBatch, const float learningRate, const DataType* testData)
	{
		const DataType& allData = trainData;
		const unsigned numBatches = allData.size() / samplesPerBatch;

		for(unsigned epoch = 0; epoch < numEpochs; ++epoch)
		{
			//group data into batches
			List<DataType> allBatches;
			for(unsigned batch = 0; batch < numBatches; ++batch)
			{
				allBatches.push_back(DataType());
				for(unsigned sampleIndex = 0; sampleIndex < samplesPerBatch; ++sampleIndex)
				{
					int batchOffset = batch * samplesPerBatch;//skip past the samples that we have already done.
					auto& sample = allData[batchOffset + sampleIndex];
					allBatches[batch].push_back(sample);
				}
			}

			//learn
			for(unsigned batch = 0; batch < allBatches.size(); ++batch)
			{
				updateMiniBatch(allBatches[batch], learningRate);
			}

			if(testData != nullptr)
			{
				List<ActiveType> guesses;
				guesses.resize(testData->size());
				for(int i = 0; i < testData->size(); ++i)
				{
					guesses[i] = (*testData)[i].first;
					feedForward(&guesses[i], biases, weights, nullptr, nullptr);
				}
				cout << "\n" << evaluate(guesses, *testData);
			}


			cout << "\nCompleted Epoch " << epoch;
		}
	}

	void updateMiniBatch(const DataType& batch, float learningRate)
	{

		//set namblas to zero
		BiasType nambla_b = biases;
		WeightType nambla_w = weights;
		initNambla(&nambla_b, &nambla_w);

		//for each sample in minibatch, compute deltas
		for(unsigned sample = 0; sample < batch.size(); ++sample)
		{
			BiasType deltaBiases = nambla_b;
			WeightType deltaWeights = nambla_w;
			backprop(batch[sample].first, batch[sample].second, &deltaBiases, &deltaWeights);


			for(unsigned layer = 0; layer < deltaBiases.size(); ++layer)
				nambla_b[layer] += deltaBiases[layer];//add to the total changes in weights and biases

			for(unsigned layer = 0; layer < deltaWeights.size(); ++layer)
				nambla_w[layer] += deltaWeights[layer];
		}

		float learnRateModifier = learningRate / batch.size();

		//update weights and biases
		for(unsigned layer = 0; layer < nambla_b.size(); ++layer)
		{
		//	if(layer == 0)
		//		cout << "\n\n\n\n" << biases[layer][0];
			biases[layer] -= learnRateModifier * nambla_b[layer];
			//cout << "\n\n\n\n" << learnRateModifier * nambla_b[layer];
		}

		for(unsigned layer = 0; layer < weights.size(); ++layer)
		{


			weights[layer] -= learnRateModifier * nambla_w[layer];
			//cout << "\n\n\n\n" << learnRateModifier * nambla_w[layer];
		}

	}
	/// <summary>
	/// Returns nambla_b and nambla_w
	/// </summary>
	void backprop(const ActiveType& inputImage, const AnswerType answer, BiasType* nambla_b_ptr, WeightType* nambla_w_ptr)
	{
		BiasType& nambla_bs = *nambla_b_ptr;
		WeightType& nambla_ws = *nambla_w_ptr;
		initNambla(&nambla_bs, &nambla_ws);

		List<ActiveType> activationPerLayer;//aka a's
		List<ActiveType> weightedInputPerLayer;//aka z's

		//store all activations layer by layer (feed forward)
		{
			ActiveType initActivation = inputImage;
			feedForward(&initActivation, biases, weights, &activationPerLayer, &weightedInputPerLayer);
			cout << "\n" << activationPerLayer.end()[-1][0];
		}

		//BP1
		ActiveType lhs = costDerivative(activationPerLayer.end()[-1], answer);
		ActiveType rhs = weightedInputPerLayer.end()[-1].unaryExpr(&sigmoidPrime);
		ActiveType delta(layerSizes.end()[-1]);
		for(int i = 0; i < layerSizes.end()[-1]; ++i)
			delta[i] = lhs[i] * rhs[i];


		//BP3
		nambla_bs.end()[-1] = delta;


		bp4(delta, activationPerLayer.end()[-2], &nambla_ws.end()[-1]);//size 10

		for(unsigned layer = 2; layer < layerSizes.size(); ++layer)
		{
			auto& z = weightedInputPerLayer.end()[-layer];
			auto& sp = z.unaryExpr(&sigmoid);

			const MatrixXf& weight = weights.end()[-layer + 1];//10x30
			MatrixXf& nambla_w = nambla_ws.end()[-layer + 1];
			ActiveType deltaNew = ActiveType(layerSizes.end()[-layer]);

			for(unsigned col = 0; col < nambla_w.cols(); ++col)
				deltaNew[col] = weight.col(col).dot(delta);
			delta = deltaNew;

			nambla_bs.end()[-layer] = delta;

			bp4(delta, activationPerLayer.end()[-layer-1], &nambla_ws.end()[-layer]);//size 10
		}


		//TODO implement //BP2
	}


	/// <summary>
	/// BP4. Should produce (d.size, a.size) matrix.
	/// </summary>
	/// <param name="activationLastLayer">a</param>
	/// <param name="delta">d</param>
	/// <param name="nambla_w_pos">Computed delta nambla weights for a layer.</param>
	void bp4(const ActiveType& delta, const ActiveType& activationLastLayer, MatrixXf* nambla_w_pos)
	{
		//BP4, check notes for example numpy code
		MatrixXf& back = *nambla_w_pos;
		for(unsigned col = 0; col < activationLastLayer.size(); ++col)
		{
			for(unsigned row = 0; row < delta.size(); ++row)
			{
				back(row, col) = activationLastLayer[col] * delta[row];
			}
		}
	}

	void initNambla(BiasType* nambla_b, WeightType* nambla_w)
	{
		(*nambla_b) = biases;
		(*nambla_w) = weights;


		for(unsigned i = 0; i < (*nambla_b).size(); ++i)
		{
			(*nambla_b)[i].setZero();
		}
		for(unsigned i = 0; i < (*nambla_w).size(); ++i)
		{
			(*nambla_w)[i].setZero();
		}
	}

	float evaluate(List<ActiveType> guesses, const DataType& answers)
	{
		int totalAttempts = guesses.size();
		int correctGuesses = 0;
		for(unsigned i = 0; i < guesses.size(); ++i)
		{
			AnswerType guess = static_cast<int>(guesses[i].col(0).maxCoeff());

			if(guess == answers[i].second)
			{
				++correctGuesses;
			}
		}

		return correctGuesses / static_cast<float>(totalAttempts);
	}

	ActiveType costDerivative(const ActiveType& guess, const AnswerType answer)
	{
		return (guess - getAnswerMat().col(answer));//TODO: this is different in the notes! WTF. Ill assume the code is correct and the notes aren't
	}
};

int main(int argc, char* argv[])
{
	Network network;
	Data data;

	DataType d;
	Eigen::VectorXf a = ActiveType(1);
	Eigen::VectorXf b = ActiveType(1);
	a[0] = 0;
	b[0] = 1;

	for(int i = 0; i < 400; ++i)
	{
		d.push_back(Pair<ActiveType, AnswerType>(a, 0));
		d.push_back(Pair<ActiveType, AnswerType>(b, 1));
	}
	

	//if(data.loadData() == false)
	//	return 1;

	//network.train(data.getTrainData(), 30, 10, 3.f, &data.getTestData());
	network.train(d, 30, 10, 3.f, &d);


	//data.getTrainImage(45, &greyImage, &sln);

	//n.feedForward(&greyImage, n.biases, n.weights);


	//cout << endl;
	//cout << endl << greyImage;
	//cout << endl;

	int i;
	cin >> i;
	return 0;
}

