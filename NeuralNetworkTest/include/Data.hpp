#pragma once

#include "stdafx.hpp"
#include "Constants.hpp"

class Data
{
public:

	Data();
	/// <summary>
	/// Returns relative path to the content directory.
	/// </summary>
	static const std::string& getContentDir();
	/// <summary>
	/// Starts from content directory. Expects file without extension, Eigen vector appropriately sized.
	/// </summary>
	static void loadImage(std::string filename, ImageType* greyImage, int* solution);


	/// <summary>
	/// Loads data from database.
	/// </summary>
	bool loadData();

	void getTrainImage(int index, ImageType* greyImage, int* solution) const;

	void getTestImage(int index, ImageType* greyImage, int* solution) const;

	const DataType& getTestData() const;
	const DataType& getTrainData() const;

	mnist::MNIST_dataset<unsigned char, unsigned char> dataBase;
private:


	mutable DataType* convertedTest;
	mutable DataType* convertedTrain;

	void getStuff(int index, ImageType* greyImage, const List<unsigned char>& image) const;
};

