#include "stdafx.hpp"
#include "Network.hpp"

float sigmoid(float z)
{
	return(1.f / (1.0f + std::powf(Constants::e, -z)));
}
float sigmoidPrime(float z)
{
	return sigmoid(z) * (1 - sigmoid(z));
}
float crossEntropy(float z)
{
	return 1.f;//still to do
}
float crossEntropyPrime(float z)
{
	return(crossEntropy(z) *(1 - crossEntropy(z)));
}
float randy(float dummy)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> d(0, 1);
	float rand = d(gen);
	return rand;
}



const MatrixXf& Network::getAnswerMat() const
{
	return answerMatrix;
}
Network::Network()
{
	init(biases, weights);
}
void Network::init(std::vector<ActiveType>& biases, std::vector<ActiveType>& weights)
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

		VectorXf biasLayer = VectorXf(layerSizes[i]);//give it a bias
		//biasLayer.setRandom();
		biasLayer = biasLayer.unaryExpr(&randy);
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
		weights[layer] = weights[layer].unaryExpr(&randy);
	}
}
void Network::feedForward(ActiveType* activation, const BiasType& biases, const WeightType& weights, List<ActiveType>* recordActivations, List<ActiveType>* recordZ)
{

	if(recordActivations)
	{
		recordActivations->resize(layerSizes.size());
		recordZ->resize(layerSizes.size() - 1);
		(*recordActivations)[0] = (*activation);
	}

	for(unsigned int layer = 0; layer < biases.size(); ++layer)
	{
		auto bias = biases[layer];
		auto weight = weights[layer];

		(*activation) = weight * (*activation) + bias;// z (weighted input)

		if(recordZ)
			(*recordZ)[layer] = (*activation);

		(*activation) = (*activation).unaryExpr(&sigmoid);// activation

		if(recordActivations)
			(*recordActivations)[layer + 1] = (*activation);
	}
}
void Network::train(const DataType& trainData, const unsigned numEpochs, const unsigned samplesPerBatch, const float learningRate, const DataType* testData)
{
	const DataType& allData = trainData;
	const unsigned numBatches = allData.size() / samplesPerBatch;

	for(unsigned epoch = 0; epoch < numEpochs; ++epoch)
	{
		//group data into batches
		List<DataType> allBatches;
		allBatches.resize(numBatches);
		for(unsigned batch = 0; batch < numBatches; ++batch)
		{
			//TODO, randomize image order to improve learning

			allBatches[batch].resize(samplesPerBatch);
			for(unsigned sampleIndex = 0; sampleIndex < samplesPerBatch; ++sampleIndex)
			{
				int batchOffset = batch * samplesPerBatch;//skip past the samples that we have already done.
				auto& sample = allData[batchOffset + sampleIndex];
				allBatches[batch][sampleIndex] = sample;
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
			std::cout << "\n" << evaluate(guesses, *testData);
		}


		std::cout << "\nCompleted Epoch " << epoch;
	}
}
void Network::updateMiniBatch(const DataType& batch, float learningRate)
{

	//set namblas to zero
	BiasType nambla_b = biases;
	WeightType nambla_w = weights;

	initNambla(&nambla_b, &nambla_w);

	BiasType deltaBiases = nambla_b;
	WeightType deltaWeights = nambla_w;

	//for each sample in minibatch, compute deltas
	for(unsigned sample = 0; sample < batch.size(); ++sample)
	{
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
void Network::backprop(const ActiveType& inputImage, const AnswerType answer, BiasType* nambla_b_ptr, WeightType* nambla_w_ptr)
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
	}

	//BP1
	ActiveType lhs = costDerivative(activationPerLayer.end()[-1], answer);
	ActiveType rhs = weightedInputPerLayer.end()[-1].unaryExpr(&sigmoidPrime);
	ActiveType delta(layerSizes.end()[-1], 1);
	delta = lhs.cwiseProduct(rhs);

	//BP3
	nambla_bs.end()[-1] = delta;
	//BP4
	bp4(delta, activationPerLayer.end()[-2], &nambla_ws.end()[-1]);//size 10

	for(unsigned layer = 2; layer < layerSizes.size(); ++layer)
	{
		auto& z = weightedInputPerLayer.end()[-layer];
		auto sigmoidPrime = z.unaryExpr(&sigmoid);

		const MatrixXf& weight = weights.end()[-layer + 1];//10x30
		MatrixXf& nambla_w = nambla_ws.end()[-layer + 1];

		//BP2
		delta = (weight.transpose() * delta).cwiseProduct(sigmoidPrime);

		nambla_bs.end()[-layer] = delta;
		//BP4
		bp4(delta, activationPerLayer.end()[-layer - 1], &nambla_ws.end()[-layer]);//size 10
	}
}
void Network::bp4(const ActiveType& delta, const ActiveType& activationLastLayer, MatrixXf* nambla_w_pos)
{
	(*nambla_w_pos) = delta * activationLastLayer.transpose();
}
void Network::initNambla(BiasType* nambla_b, WeightType* nambla_w)
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
float Network::evaluate(List<ActiveType> guesses, const DataType& answers)
{
	int totalAttempts = guesses.size();
	int correctGuesses = 0;
	for(unsigned i = 0; i < guesses.size(); ++i)
	{
		AnswerType guess;
		guesses[i].col(0).maxCoeff(&guess);//get the index of the highest value

		if(guess == answers[i].second)
		{
			++correctGuesses;
		}
		else
		{
			//TODO: save bad guesses and look at what they looked like
		}
	}

	return correctGuesses / static_cast<float>(totalAttempts);
}
ActiveType Network::costDerivative(const ActiveType& guess, const AnswerType answer)
{
	return (guess - getAnswerMat().col(answer));
}

