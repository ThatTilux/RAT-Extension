#include "objective_function.h"

ObjectiveFunction::ObjectiveFunction(const ModelHandler &model_handler, double weight_chisquared)
    : model_handler_(model_handler),
      json_file_path_(model_handler.getTempJsonPath()),
      calculator_(model_handler.getTempJsonPath()),
      weight_chisquared_(weight_chisquared)
{
    // check that there are only custom harmonics with an 'amplitude' of linear
    HarmonicDriveParameterMap params = model_handler_.getHarmonicDriveValues();
    for (auto &param : params)
    {
        if (!param.second.isOffsetAndSlope())
            throw std::runtime_error("The selected model has one or more custom harmonics with an 'amplitude' value other than 'linear'. This is not supported for this optimizer.");
    }
}

// Objective function for Bayesian Optimization. Incorporates the sum of the bn values (except the main one) and chi-square differences to a fitted linear function
// A lower value is better, 0 is the minimum
// There are 18 input params: offset and slope for all customs harmonics B1...B10 except for the main one
double ObjectiveFunction::objective_function(const HarmonicDriveParameterMap &params)
{
    // TODO select correct component (incase it is not the 10,000 one)
    // TODO take absolute of bn values
    // also investigate if chiSquared can be negative

    // apply all paramaters
    model_handler_.apply_params(params);

    // do the computation
    HarmonicsHandler harmonics_handler;
    calculator_.reload_and_calc(json_file_path_, harmonics_handler);
    std::vector<double> current_bn_values = harmonics_handler.get_bn();

    // get the sum of all bns except for the 10,000 one
    double sum_bn = std::accumulate(current_bn_values.begin(), current_bn_values.end(), 0.0) - 10000;

    // sum up all chi squared values except for the main one
    double sum_chisquared = 0;
    for (int i = 1; i <= 10; i++)
    {
        if (current_bn_values[i - 1] != 10000)
        {
            sum_chisquared += chiSquared(harmonics_handler, i);
        }
    }

    // compute the objective function
    double objective_value = sum_bn + weight_chisquared_ * sum_chisquared;

    return objective_value;
}

// Function to compute chi-squared distance between (1) a function described by the points vector and (2) a linear function described by slope, intercept
double computeChiSquared(const std::vector<std::pair<double, double>> &points, double slope, double intercept, double variance_y)
{
    double chi_squared = 0;
    for (const auto &point : points)
    {
        double y_fit = slope * point.first + intercept;
        double residual = point.second - y_fit;
        chi_squared += (residual * residual) / variance_y;
    }
    return chi_squared;
}

// Function to compute the chi square for a Bn component. A linear function will be fitted to the data and chi square will be computed between that and the data
double chiSquared(HarmonicsHandler &harmonics_handler, int component)
{
    // get the Bn and ell
    std::vector<double> Bn = harmonics_handler.get_Bn(component);
    std::vector<double> ell = harmonics_handler.get_ell();

    // get the variance of Bn
    double variance = computeVariance(Bn);

    // stitch ell and Bn together
    std::vector<std::pair<double, double>> points = combinePoints(ell, Bn);

    // fit a linear function
    auto [slope, intercept] = linearRegression(points);

    // TODO TEMP START --------------------

    std::cout << "Slope: " << slope << ", Intercept: " << intercept << std::endl;

    // TODO TEMP END  --------------------

    // compute chi squared between the function and the original data
    double chi_squared = computeChiSquared(points, slope, intercept, variance);

    return chi_squared;
}

// Function to perform linear regression and return the slope and intercept
std::pair<double, double> linearRegression(const std::vector<std::pair<double, double>> &points)
{
    size_t n = points.size();
    if (n < 2)
    {
        throw std::runtime_error("Not enough points for linear regression.");
    }

    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (const auto &point : points)
    {
        sum_x += point.first;
        sum_y += point.second;
        sum_xx += point.first * point.first;
        sum_xy += point.first * point.second;
    }

    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;

    return {slope, intercept};
}

// Function to compute the variance of y values with Bessel's correction
double computeVariance(const std::vector<double> &y)
{
    double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double variance = 0;
    for (const auto &value : y)
    {
        variance += (value - mean) * (value - mean);
    }
    variance /= (y.size() - 1); // Bessel's correction
    return variance;
}