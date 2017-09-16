#include "Data.hpp"
#include "LodePng.hpp"


std::string mnist::filePath = Data::getContentDir() + "mnist/";

Data::Data()
{

}
Data::~Data()
{

}


std::string Data::getContentDir()
{
	return "../content/";
}
void Data::loadPngAsGrayscale(std::string filename, ImageVector* greyImage, int* solution)
{
	std::vector<unsigned char> rgbaImage;
	unsigned width, height;
	unsigned error = lodepng::decode(rgbaImage, width, height, Data::getContentDir() + filename);

	if(error)
		std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	//image now in RGBA, RGBA etc.

	//convert to greyscale
	for(unsigned int i = 0; i < rgbaImage.size(); i += 4)
	{
		(*greyImage)[i / 4] = static_cast<Pixel>((rgbaImage[i] + rgbaImage[i + 1] + rgbaImage[i + 2]) / 3);
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