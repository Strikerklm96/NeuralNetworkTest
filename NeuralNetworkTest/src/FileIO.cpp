#include "FileIO.hpp"

std::string File::getContentDir()
{
	return "../content/";
}
void File::loadPngAsGrayscale(std::string filename, Eigen::VectorXf* greyImage)
{
	std::vector<unsigned char> rgbaImage;
	unsigned width, height;
	unsigned error = lodepng::decode(rgbaImage, width, height, File::getContentDir() + filename);

	if(error)
		std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	//image now in RGBA, RGBA etc.

	//convert to greyscale
	for(unsigned int i = 0; i < rgbaImage.size(); i += 4)
	{
		(*greyImage)[i / 4] = static_cast<float>((rgbaImage[i] + rgbaImage[i + 1] + rgbaImage[i + 2]) / 3);
	}
}
