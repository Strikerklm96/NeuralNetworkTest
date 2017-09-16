
#include "stdafx.hpp"

using namespace std;
using namespace Eigen;





int main(int argc, char* argv[])
{
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

