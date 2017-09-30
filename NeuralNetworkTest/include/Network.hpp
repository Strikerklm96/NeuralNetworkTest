#pragma once

#include "stdafx.hpp"
#include "Constants.hpp"

using namespace Eigen;

float sigmoid(float z);
float sigmoidPrime(float z);

float crossEntropy(float z);
float crossEntropyPrime(float z);
float randy(float dummy);

class Network
{
public:

	Network();

	const int inputLayerSize = Constants::imageSize;
	const int hiddenLayerSize = 30;
	const int numOutputs = 10;
	std::vector<int> layerSizes;

	BiasType biases;//Biases modify output value for all connections.
	WeightType weights;//Weights modify value for each connection.

	const MatrixXf& getAnswerMat() const;
	void feedForward(ActiveType* activation, const BiasType& biases, const WeightType& weights, List<ActiveType>* recordActivations, List<ActiveType>* recordZ);
	void train(const DataType& trainData, const unsigned numEpochs, const unsigned samplesPerBatch, const float learningRate, const DataType* testData);


private:
	ActiveType preprocess(const ActiveType& inputImage);
	MatrixXf answerMatrix;
	void updateMiniBatch(const DataType& batch, float learningRate);
	void backprop(const ActiveType& inputImage, const AnswerType answer, BiasType* nambla_b_ptr, WeightType* nambla_w_ptr);
	void bp4(const ActiveType& delta, const ActiveType& activationLastLayer, MatrixXf* nambla_w_pos);
	void init(std::vector<ActiveType>& biases, std::vector<ActiveType>& weights);
	void initNambla(BiasType* nambla_b, WeightType* nambla_w);

	float evaluate(List<ActiveType> guesses, const DataType& answers);

	ActiveType costDerivative(const ActiveType& guess, const AnswerType answer);
};

