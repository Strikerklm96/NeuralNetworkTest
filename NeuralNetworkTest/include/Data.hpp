#pragma once

#include "stdafx.hpp"
#include "Constants.hpp"

class Data
{
public:
	/// <summary>
	/// Returns relative path to the content directory.
	/// </summary>
	static const std::string& getContentDir();
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

	mnist::MNIST_dataset<unsigned char, unsigned char> dataBase;
private:

	void getStuff(int index, ImageVector* greyImage, const std::vector<unsigned char>& image);
};

