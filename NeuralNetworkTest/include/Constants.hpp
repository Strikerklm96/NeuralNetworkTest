#pragma once

#include "stdafx.hpp"

class Constants
{
public:
	const static int imageSize = 28 * 28;
	Constants();
	~Constants();
};


typedef Eigen::Matrix < float, Constants::imageSize, 1 > ImageVector;
