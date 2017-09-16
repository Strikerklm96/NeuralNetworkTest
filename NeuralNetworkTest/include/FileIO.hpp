#pragma once

#include "stdafx.hpp"
#include "LodePng.hpp"

/// <summary>
/// Handles file IO.
/// </summary>
class File
{
public:
	/// <summary>
	/// Returns relative path to the content directory.
	/// </summary>
	static std::string getContentDir();
	/// <summary>
	/// Starts from content directory. Expects file name and Eigen vector appropriately sized.
	/// </summary>
	static void loadPngAsGrayscale(std::string filename, Eigen::VectorXf* greyImage);
};

