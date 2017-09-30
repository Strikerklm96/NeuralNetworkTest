
#include "Constants.hpp"
#include "Data.hpp"
#include "Network.hpp"
#include <Windows.h>

#include "FreeImage.h"
#include "FreeImagePlus.h"

using namespace std;
using namespace Eigen;

#define WIDTH 800
#define HEIGHT 600
#define BPP 24 

volatile int stop;
void cinListen()
{
	int t;
	cin >> t;
	stop = t;
}
int main(int argc, char* argv[])
{

	fipImage image;
	image.load("../content/image1.png");
	image.rotate(-90);

	image.save("../content/testImage.png");

	ActiveType t(4, 1);
	t.setZero();
	t(1, 0) = 1;


	std::thread stopThread(cinListen);

	////Example bp4 fix
	//MatrixXf a(3,1);
	//a(0, 0) = 1;
	//a(1, 0) = 1;
	//a(2, 0) = 1;
	//MatrixXf b(2, 1);
	//b(0, 0) = 2;
	//b(1, 0) = 3;
	//MatrixXf asdf = a.transpose();

	//MatrixXf t = b * asdf;



	Network network;
	Data data;

	//DataType d;
	//Eigen::VectorXf a = ActiveType(1);
	//Eigen::VectorXf b = ActiveType(1);
	//a[0] = 0;
	//b[0] = 1;

	//for(int i = 0; i < 400; ++i)
	//{
	//	d.push_back(Pair<ActiveType, AnswerType>(a, 0));
	//	d.push_back(Pair<ActiveType, AnswerType>(b, 1));
	//}
	//network.train(d, 30, 10, 3.f, &d);


	if(data.loadData() == false)
		return 1;

	stop = 0;
	while(!stop)
	{
		auto& datas = data.getTrainData();
		std::random_shuffle(datas.begin(), datas.end());
		network.train(datas, 1, 10, 0.5f, &data.getTestData());
	}

	int sol = 6;
	int i = 0;
	while(true)
	{
		ActiveType image(Constants::imageSize, 1);
		data.loadImage("image" + std::to_string(i), &image, &sol);
		network.feedForward(&image, network.biases, network.weights, nullptr, nullptr);
		AnswerType guess;
		image.col(0).maxCoeff(&guess);
		cout << "\n\n===========================\nGuess was " << guess << "\nWith Values\n" << image << "\n\n Pick Next Image:";
		cin >> i;
	}

	stopThread.join();
	cin >> i;
	return 0;
}

