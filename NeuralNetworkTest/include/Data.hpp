#pragma once

#include "stdafx.hpp"
#include "Constants.hpp"

class Data
{
public:
	Data();
	~Data();


	/// <summary>
	/// Returns relative path to the content directory.
	/// </summary>
	static std::string getContentDir();
	/// <summary>
	/// Starts from content directory. Expects file without extension, Eigen vector appropriately sized.
	/// </summary>
	static void loadImage(std::string filename, ImageVector* greyImage, int* solution);


	/// <summary>
	/// Loads data from database.
	/// </summary>
	bool loadData();

	void getTrainImage(int index, ImageVector* greyImage, int* solution);

	void getTestImage(int index, ImageVector* greyImage, int* solution);

private:
	mnist::MNIST_dataset<Pixel, Pixel> dataBase;

};

