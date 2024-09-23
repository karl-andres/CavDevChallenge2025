#include "matrix_utils.h"

VectorXd rollVector(VectorXd& values, int numElements) {

    // copy input vector into vector rolled
    VectorXd rolled(values);
    int len = (int)values.rows();

    // get the last n entries of values and set that to top n entries of rolled
    // get the first len - n entries of values and set that to bottom len - n entries of rolled
    auto tempNewTop = values.bottomRows(numElements);
    rolled.bottomRows(len - numElements) = values.topRows(len - numElements);
    rolled.topRows(numElements) = tempNewTop;

    return rolled;

}

VectorXd getFirstDifferences(VectorXd& values) {

    // subtract values from itself rolled forward 1 to get differences
    VectorXd differences = values - rollVector(values, 1);

    // set first entry to 0 (will be garbage value with the roll operation)
    differences(0) = 0;

    return differences;

}

VectorXd cumsum(VectorXd &values) {

    // initialize cumsum vector with zeros
    VectorXd cumsum_values = VectorXd::Zero(values.rows());

    // calculate the rolling sum for each index i and set cumsum vector to the sum up to that index
    double sum = 0;
    for (int i = 0; i < values.rows(); i++) {
        sum += values(i);
        cumsum_values(i) = sum;
    }

    return cumsum_values;

}

int binarySearchRanged(VectorXd& values, double input) {

    // explicitly checked assumptions: values vector has at least two entries
    // UNCHECKED ASSUMPTION: values is sorted in non-decreasing order
    assert(values.rows() >= 2);

    // edge case: if input is greater than or equal to the last element of the array, return the length of the array
    if (input >= values(values.rows() - 1)) { return values.rows() - 1; }

    // edge case: if input is less than first value, return -1
    if (input < values(0)) { return -1; }

    int low = 0, high = (int)(values.rows() - 2);
    while (true) {

        int idx = (int)((low + high) / 2);

        // binary search to find lower index of range of values input value falls between
        if (values(idx) <= input && input < values(idx+1)) { return idx; }
        else if (input < values(idx)) { high = idx; }
        else { low = idx + 1; } 

    }

}

double interpolate1d(VectorXd& inputs, VectorXd& outputs, double input) {

    // get lower index of range input falls between
    int lower_index = binarySearchRanged(inputs, input);

    // if lower index is the last index, return the final output
    if (lower_index == outputs.rows() - 1) { return outputs(outputs.rows() - 1); }

    // if lower index is -1 (input is lower than content of inputs), return lowest value
    if (lower_index == -1) { return outputs(0); }

    // otherwise, linear interpolate between values
    // r = (x - x1)/(x2 - x1)
    // y = y1 + r*(y2 - y1)
    double ratio = (input - inputs(lower_index)) / (inputs(lower_index + 1) - inputs(lower_index));
    return (outputs(lower_index) + ratio*(outputs(lower_index + 1) - outputs(lower_index)));

}

Matrix2d getRotMat(double heading_rad) {

    Matrix2d rotMat {
        {cos(heading_rad), -sin(heading_rad)},
        {sin(heading_rad), cos(heading_rad)}
    };

    return rotMat;

}