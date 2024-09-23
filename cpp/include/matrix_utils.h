#pragma once

#include <Eigen/Dense>

using namespace Eigen;

/**
@brief given a vector of values, roll the entries forward by a specified amount
@param values reference to vector to roll
@param numElements number of elements to roll forward
@return copy of initial vector with values shifted
*/
VectorXd rollVector(VectorXd& values, int numElements);

/**
@brief given a vector of values, return the first differences of the values
@param values reference to Eigen vector of values to take differences of
@return vector of first differences of the same length as the original vector, first entry padded with 0
*/
VectorXd getFirstDifferences(VectorXd& values);

/**
@brief calculates and returns rolling sum of a vector
@param values reference to input vector
@return Eigen copied vector of rolling sum
*/
VectorXd cumsum(VectorXd& values);

/**
@brief linearly interpolates between a vector of inputs to a vector of outputs at specified input
@param inputs input vector for interpolation
@param outputs output vector for interpolation
@param input input to interpolate
@return interpolated value
*/
double interpolate1d(VectorXd& inputs, VectorXd& outputs, double input);

/**
@brief get rotation matrix from heading angle
@param heading_rad angle in radians
*/
Matrix2d getRotMat(double heading_rad);