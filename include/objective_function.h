#ifndef OBJECTIVE_FUNCTION_H
#define OBJECTIVE_FUNCTION_H

#include <vector>
#include "harmonics_calculator.h"
#include "harmonics_handler.h"
#include "model_handler.h"
#include "constants.h"
#include <iostream>
#include <cmath>


class ObjectiveFunction{    
    public:
        ObjectiveFunction(const boost::filesystem::path &json_file_path, const ModelHandler &model_handler);

        double objective_function(const std::unordered_map<std::string, HarmonicDriveParameters> &params, double weight_chisquared);
        
    private:

        boost::filesystem::path json_file_path_;
        HarmonicsCalculator calculator_;
        ModelHandler model_handler_;
};


double chiSquared(HarmonicsHandler &harmonics_handler, int component);
double computeVariance(const std::vector<double> &y);
std::pair<double, double> linearRegression(const std::vector<std::pair<double, double>> &points);


#endif // OBJECTIVE_FUNCTION_H