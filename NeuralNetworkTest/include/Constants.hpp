#pragma once

#include "stdafx.hpp"

class Constants
{
public:
	const static int imageSize = 28 * 28;

	static Eigen::VectorXf ImageVector()
	{
		return Eigen::VectorXf(Constants::imageSize);
	}

	Constants();
	~Constants();
};


