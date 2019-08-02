/**
 * @file rf_api.hpp
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


#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>

namespace mlpack {
namespace tree {

class RandomForestAPI 
{
  public:

  /**
   * Classify a given set of data using a previously trained random forest. Also
   * returns the prediction probabilities for each data point.
   *
   * @param model_file Filename of previously trained random forest used to classify data.
   * @param data Dataset to be classified.
   * @param predictions Vector to store predictions on dataset.
   * @param probabilities Output matrix of class probabilities for each point.
   */
  void mlpack_rf_test(const char* model_file, const MatType& data, arma::Row<size_t>& predictions, arma::mat& probabilities)


  /**
   * Clean up function to manage left over data from classification.
   *
   * @param model_file Filename for previously opened model file to now close/destroy.
   * @param data Dataset that was used to classify.
   * @param predictions Vector previously used to store predictions.
   * @param probabilities Probability matrix used to store probabilites of each prediction.
   */
  void mlpack_cleanup(const char* model_file, const MatType& data, arma::Row<size_t>& predictions, arma::mat& probabilities)


};

} // namespace tree
} // namespace mlpack
