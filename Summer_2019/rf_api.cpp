/**
 * @file rf_api.cpp
 * @author Connor Smith
 *
 * Acts as an interface between mlpack's random forest algorithm and any 
 * application that utilizes random forest capabilities
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


#include <rf_api.hpp>

extern "C++" {
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
}


using namespace mlpack;
using namespace mlpack::tree;


void mlpack_rf_classify(std::string model_file, std::string data, std::vector<int> predictions, std::vector<float> probabilities) { 
  arma::mat test_data;
  arma::mat probabilities;
  arma::Row<size_t> predictions;

  //TODO: load model_file from csv

  RandomForest<> rf;

  //RandomForestModel* rfModel = model_file;

  rfModel->rf.Classify(test_data, predictions, probabilities);

}

void mlpack_rf_cleanup(std::string model_file, std::string data, std::vector<int> predictions, std::vector<float> probabilities) { } 
