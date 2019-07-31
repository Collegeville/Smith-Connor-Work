// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#include "MueLu_AvatarInterface.hpp"

#include <string> 
#include <fstream> 
#include <sstream> 
#include <vector> 
#include "Teuchos_Array.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "MueLu_BaseClass.hpp"
#include "Teuchos_RawParameterListHelpers.hpp"


// ***********************************************************************
/* Notional Parameterlist Structure
   "mlpack: model files"   	   "{'mymodel1.bin','mymodel2.bin'}"
   "mlpack: good class"            "1"
   "mlpack: heuristic" 		   "1"
   "mlpack: bounds file"           "{'bounds.data'}"
   "mlpack: muelu parameter mapping"
     - "param0'
       - "muelu parameter"          "aggregation: threshold"
       - "mlpack parameter"         "DROP_TOL"
       - "muelu values"             "{0,1e-4,1e-3,1e-2}"
       - "mlpack values"            "{1,2,3,4}
     - "param1'
       - "muelu parameter"          "smoother: sweeps"
       - "mlpack parameter"         "SWEEPS"
       - "muelu values"             "{1,2,3}"
       - "mlpack values"            "{1,2,3}"


   Notional SetMueLuParameters "problemFeatures"  Structure
   "my feature #1"                "246.01"
   "my feature #2"                "123.45"

 */


#ifdef HAVE_MUELU_MLPACK

// ***********************************************************************
RCP<const ParameterList> MLpackInterface::GetValidParameterList() const {
  RCP<ParameterList> validParamList = rcp(new ParameterList());

  Teuchos::ParameterList pl_dummy;
  Teuchos::Array<std::string> ar_dummy;
  int int_dummy;

  // Files from which to load mlpack model
  validParamList->set<Teuchos::Array<std::string> >("mlpack: model files",ar_dummy,"Names of MLpack model files");

  // This should be a MueLu parameter-to-Avatar parameter mapping (e.g. if mlpack doesn't like spaces)
  validParamList->set<Teuchos::ParameterList>("mlpack: muelu parameter mapping",pl_dummy,"Mapping of MueLu to MLpack Parameters");

  // "Good" Class ID for mlpack
  validParamList->set<int>("mlpack: good class",int_dummy,"Numeric code for class MLpack considers to be good");

   // Which drop tol choice heuristic to use
  validParamList->set<int>("mlpack: heuristic",int_dummy,"Numeric code for which heuristic we want to use");  

  // Bounds file for extrapolation risk
  validParamList->set<Teuchos::Array<std::string> >("mlpack: bounds file",ar_dummy,"Bounds file for MLpack extrapolation risk");

  // Add dummy variables at the start
  validParamList->set<int>("mlpack: initial dummy variables",int_dummy,"Number of dummy variables to add at the start");

  // Add dummy variables before the class
  validParamList->set<int>("mlpack: pre-class dummy variables",int_dummy,"Number of dummy variables to add at the before the class");

  return validParamList;
}



// ***********************************************************************
Teuchos::ArrayRCP<std::string> MLpackInterface::ReadFromFiles(const char * paramName) const {
  //  const Teuchos::Array<std::string> & tf = params_.get<const Teuchos::Array<std::string> >(paramName);
  Teuchos::Array<std::string> & tf = params_.get<Teuchos::Array<std::string> >(paramName);
  Teuchos::ArrayRCP<std::string> treelist;
  // Only Proc 0 will read the files and print the strings
  if (comm_->getRank() == 0) {
    treelist.resize(tf.size());
    for(Teuchos_Ordinal i=0; i<tf.size(); i++) {
      std::fstream file;
      std::stringstream ss;
      file.open(tf[i]);
      ss << file.rdbuf();
      treelist[i] = ss.str();
      file.close();
    }
  }
  return treelist;
}



// ***********************************************************************
void MLpackInterface::Setup() {
  // Sanity check
  if(comm_.is_null()) throw std::runtime_error("MueLu::MLpackInterface::Setup(): Communicator cannot be null");

  // Get the avatar strings (NOTE: Only exist on proc 0)
  mlpackStrings_ = params_.get<Teuchos::Array<std::string>>("mlpack: model files");
  if(params_.isParameter("mlpack: bounds file"))
    boundsString_ = ReadFromFiles("mlpack: bounds file");

  // Which class does MLpack consider "good"
  mlpackGoodClass_ = params_.get<int>("mlpack: good class");

  heuristicToUse_ = params_.get<int>("mlpack: heuristic");

  // Unpack the MueLu Mapping into something actionable
  UnpackMueLuMapping();

}


// ***********************************************************************
void MLpackInterface::GenerateFeatureString(const Teuchos::ParameterList & problemFeatures, std::string & featureString) const {
  // NOTE: Assumes that the features are in the same order MLpack wants them.
  std::stringstream ss;

  // Initial Dummy Variables
  if (params_.isParameter("mlpack: initial dummy variables")) {
    int num_dummy = params_.get<int>("mlpack: initial dummy variables");
    for(int i=0; i<num_dummy; i++)
      ss<<"666,";
  }

  for(Teuchos::ParameterList::ConstIterator i=problemFeatures.begin(); i != problemFeatures.end(); i++) {
    //    const std::string& name = problemFeatures.name(i);
    const Teuchos::ParameterEntry& entry = problemFeatures.entry(i);
    if(i!=problemFeatures.begin()) ss<<",";
    entry.leftshift(ss,false);  // Because ss<<entry prints out '[unused]' and we don't want that.
  }
  featureString = ss.str();
}



// ***********************************************************************
void MLpackInterface::UnpackMueLuMapping() {
  const Teuchos::ParameterList & mapping = params_.get<Teuchos::ParameterList>("mlpack: muelu parameter mapping");
  // Each MueLu/MLpack parameter pair gets its own sublist.  These must be numerically ordered with no gap

  bool done=false; 
  int idx=0;
  int numParams = mapping.numParams();

  mueluParameterName_.resize(numParams);
  mlpackParameterName_.resize(numParams);
  mueluParameterValues_.resize(numParams);
  mlpackParameterValues_.resize(numParams);

  while(!done) {
    std::stringstream ss; 
    ss << "param" << idx;
    if(mapping.isSublist(ss.str())) {
      const Teuchos::ParameterList & sublist = mapping.sublist(ss.str());

      // Get the names
      mueluParameterName_[idx]  = sublist.get<std::string>("muelu parameter");
      mlpackParameterName_[idx] = sublist.get<std::string>("mlpack parameter");

      // Get the values
      mueluParameterValues_[idx]  = sublist.get<Teuchos::Array<double> >("muelu values");
      mlpackParameterValues_[idx] = sublist.get<Teuchos::Array<double> >("mlpack values");            

      idx++;
    }
    else {
      done=true;
    }
  }

  if(idx!=numParams) 
    throw std::runtime_error("MueLu::MLpackInterface::UnpackMueLuMapping(): 'mlpack: muelu parameter mapping' has unknown fields");
}


// ***********************************************************************
std::string MLpackInterface::ParamsToString(const std::vector<int> & indices) const {
  std::stringstream ss;
  for(Teuchos_Ordinal i=0; i<mlpackParameterValues_.size(); i++) {
    ss << "," << mlpackParameterValues_[i][indices[i]];
  }

  // Pre-Class dummy variables
  if (params_.isParameter("mlpack: pre-class dummy variables")) {
    int num_dummy = params_.get<int>("mlpack: pre-class dummy variables");
    for(int i=0; i<num_dummy; i++)
      ss<<",666";
  }
  
  return ss.str();
}



// ***********************************************************************
void MLpackInterface::SetIndices(int id,std::vector<int> & indices) const {
  // The combo numbering here starts with the first guy
  int numParams = (int)mlpackParameterValues_.size();
  int curr_id = id;
  for(int i=0; i<numParams; i++) {
    int div = mlpackParameterValues_[i].size();
    int mod = curr_id % div;
    indices[i] = mod;
    curr_id = (curr_id - mod)/div;
  }
}


// ***********************************************************************
void MLpackInterface::GenerateMueLuParametersFromIndex(int id,Teuchos::ParameterList & pl) const {
  // The combo numbering here starts with the first guy
  int numParams = (int)mlpackParameterValues_.size();
  int curr_id = id;
  for(int i=0; i<numParams; i++) {
    int div = mlpackParameterValues_[i].size();
    int mod = curr_id % div;
    pl.set(mueluParameterName_[i],mueluParameterValues_[i][mod]);
    curr_id = (curr_id - mod)/div;
  }
}



// ***********************************************************************
void MLpackInterface::SetMueLuParameters(const Teuchos::ParameterList & problemFeatures, Teuchos::ParameterList & mueluParams, bool overwrite) const {
  Teuchos::ParameterList mlpackParams;
  std::string paramString;

  if (comm_->getRank() == 0) {
    // Turn the problem features into a "trial" string to run as test values in mlpack
    std::string trialString;
    GenerateFeatureString(problemFeatures,trialString);
    
    // Compute the number of things we need to test
    int numParams = (int)mlpackParameterValues_.size();
    std::vector<int> indices(numParams);
    std::vector<int> sizes(numParams);
    int num_combos = 1;
    for(int i=0; i<numParams; i++) {
      sizes[i]    = mlpackParameterValues_[i].size();
      num_combos *= mlpackParameterValues_[i].size();
    }
    GetOStream(Runtime0)<< "MueLu::MLpackInterface: Testing "<< num_combos << " option combinations"<<std::endl;

    // For each input parameter to avatar we iterate over its allowable values and then compute the list of options which MLpack
    // views as acceptable
    // FIXME: Find method to find number of classes within mlpack
    int num_classes = 3;
    std::vector<int> predictions(num_combos, 0);
    std::vector<float> probabilities(num_classes * num_combos, 0);

      std::string testString;
      for(int i=0; i<num_combos; i++) {
        SetIndices(i,indices);
        // Now we add the MueLu parameters into one, enormous trial string and run avatar once
        testString += trialString + ParamsToString(indices) + ",0\n";
      }

      std::cout<<"** MLpack TestString ***\n"<<testString<<std::endl;//DEBUG

      int bound_check = true;
      if(params_.isParameter("mlpack: bounds file"))
         bound_check = checkBounds(testString, boundsString_);

    // *********************************      
    // FIXME: CLASSIFY TEST STRING HERE (predictions stored in predictions[] vector)
    // *********************************

    // Look at the list of acceptable combinations of options 
    std::vector<int> acceptableCombos; acceptableCombos.reserve(100);
    for(int i=0; i<num_combos; i++) {    
      if(predictions[i] == mlpackGoodClass_) acceptableCombos.push_back(i);      
    }
    GetOStream(Runtime0)<< "MueLu::MLpackInterface: "<< acceptableCombos.size() << " acceptable option combinations found"<<std::endl;

    // Did we have any good combos at all?
    int chosen_option_id = 0;
    if(acceptableCombos.size() == 0) { 
      GetOStream(Runtime0) << "WARNING: MueLu::MLpackInterface: found *no* combinations of options which it believes will perform well on this problem" <<std::endl
                           << "         An arbitrary set of options will be chosen instead"<<std::endl;    
    }
    else {
      // If there is only one acceptable combination, use it; 
      // otherwise, find the parameter choice with the highest
      // probability of success
      if(acceptableCombos.size() == 1){
	chosen_option_id = acceptableCombos[0];
      } 
      else {
	switch (heuristicToUse_){
	  case 1: 
		chosen_option_id = hybrid(probabilities.data(), acceptableCombos);
		break;
	  case 2: 
		chosen_option_id = highProb(probabilities.data(), acceptableCombos);
		break;
	  case 3: 
		// Choose the first option in the list of acceptable
		// combinations; the lowest drop tolerance among the 
		// acceptable combinations
		chosen_option_id = acceptableCombos[0];
		break;
	  case 4: 
		chosen_option_id = lowCrash(probabilities.data(), acceptableCombos);
		break;
	  case 5:
		chosen_option_id = weighted(probabilities.data(), acceptableCombos);
		break;
        }

      }
    }
    
    // If mesh parameters are outside bounding box, set drop tolerance
    // to 0, otherwise use avatar recommended drop tolerance
    if (bound_check == 0){
      GetOStream(Runtime0) << "WARNING: Extrapolation risk detected, setting drop tolerance to 0" <<std::endl;
      GenerateMueLuParametersFromIndex(0,mlpackParams);
    } else {
      GenerateMueLuParametersFromIndex(chosen_option_id,mlpackParams);
    }
  } 

  Teuchos::updateParametersAndBroadcast(outArg(mlpackParams),outArg(mueluParams),*comm_,0,overwrite);


}



int MLpackInterface::checkBounds(std::string trialString, Teuchos::ArrayRCP<std::string> boundsString) const {
  std::stringstream ss(trialString);
  std::vector<double> vect;

  double b; 
  while (ss >> b)  {
    vect.push_back(b);
    if (ss.peek() == ',') ss.ignore();
  }
  
  std::stringstream ssBounds(boundsString[0]);
  std::vector<double> boundsVect;

  while (ssBounds >> b) {
    boundsVect.push_back(b);    
    if (ssBounds.peek() == ',') ssBounds.ignore();
  }

  int min_idx = (int) std::min(vect.size(),boundsVect.size()/2);

  bool inbounds=true;
  for(int i=0; inbounds && i<min_idx; i++) 
    inbounds =  boundsVect[2*i] <= vect[i] && vect[i] <= boundsVect[2*i+1];

  return (int) inbounds;
}

