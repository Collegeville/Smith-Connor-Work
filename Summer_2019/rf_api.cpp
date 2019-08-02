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

#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>

using namespace mlpack;
using namespace mlpack::tree;


void mlpack_rf_test(const char* model_file, const MatType& data, arma::Row<size_t>& predictions, arma::mat& probabilities) { }

  void mlpack_cleanup(const char* model_file, const MatType& data, arma::Row<size_t>& predictions, arma::mat& probabilities) { } 
