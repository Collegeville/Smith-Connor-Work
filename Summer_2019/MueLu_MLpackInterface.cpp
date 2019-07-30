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






