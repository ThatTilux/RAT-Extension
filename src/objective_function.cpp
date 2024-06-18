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
// There are 18 input params: offset and slope for all customs harmonics B1...B10 except for the main one (or 9 with just the offsets incase the bn optimizer is used)
double ObjectiveFunction::objective_function(HarmonicDriveParameterMap &params)
{
    // TODO MAKE THIS MORE MODULAR 

    // TODO for simplicity, we assume that B2 is the main component
    int main_component = 2; // B2

    // apply all parameters
    model_handler_.apply_params(params);

    // if the bn optimizer should be used, apply it
    if(USE_BN_OPTIMIZER_IN_CHISQUARED){
        Logger::info("= Starting bn optimizer... =");
        // dummy vector that optimize puts the results in; we don't need this here
        std::vector<double> bn;
        //get drive values
        HarmonicDriveParameterMap harmonic_drive_values = model_handler_.getHarmonicDriveValues();
        // run the bn optimizer
        optimize(calculator_, model_handler_, bn, harmonic_drive_values, MAX_BN_CHISQUARE, model_handler_.getTempJsonPath());
        Logger::info("= bn optimizer has finished =");
    }

    // do the computation
    HarmonicsHandler harmonics_handler;
    Logger::info("Running harmonics calculation...");
    calculator_.reload_and_calc(json_file_path_, harmonics_handler);
    std::vector<double> current_bn_values = harmonics_handler.get_bn();

    // print bn values
    print_vector(current_bn_values, "bn");

    // get the sum of all absolute bns except for the main one
    double sum_bn = std::accumulate(current_bn_values.begin(), current_bn_values.end(), 0.0, [](double a, double b)
                                    { return a + std::abs(b); }) -
                    std::abs(current_bn_values[main_component - 1]);

    // distance of the main bn value to 10,000
    double distance_main_10000 = 10000 - std::abs(current_bn_values[main_component - 1]);

    // sum up all chi squared values except for the main one
    double sum_chisquared = 0;
    Logger::info("chiSquared values:");
    for (int i = 1; i <= 10; i++)
    {
        if (i != main_component)
        {
            double value = chiSquared(harmonics_handler, i);
            sum_chisquared += value;
            Logger::info("chiSquared[" + std::to_string(i) + "]: " + std::to_string(value));
        }
    }

    Logger::info("bn objective value: " + std::to_string(sum_bn));
    Logger::info("distance of main bn to 10,000: " + std::to_string(distance_main_10000));
    Logger::info("chiSquared objective value: " + std::to_string(sum_chisquared) + ", weighted: " + std::to_string(sum_chisquared * weight_chisquared_));

    // compute the objective function
    double objective_value;
    if (USE_BN_OPTIMIZER_IN_CHISQUARED){
        Logger::info("Only including chiSquared for the objective function");
        objective_value = sum_chisquared;
    } else {
        objective_value = sum_bn + distance_main_10000 + weight_chisquared_ * sum_chisquared;
    }

    Logger::info("objective function value: " + std::to_string(objective_value));

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