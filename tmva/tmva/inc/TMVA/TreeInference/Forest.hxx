/**********************************************************************************
 * Project: ROOT - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Authors:                                                                       *
 *      Stefan Wunsch (stefan.wunsch@cern.ch)                                     *
 *      Luca Zampieri (luca.zampieri@alumni.epfl.ch)                              *
 *                                                                                *
 * Copyright (c) 2019:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_TREEINFERENCE_FOREST
#define TMVA_TREEINFERENCE_FOREST

#include <functional>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include "TFile.h"
#include "TDirectory.h"

#include "BranchlessTree.hxx"
#include "Objectives.hxx"

namespace TMVA {
namespace Experimental {

namespace Internal {
template <typename T>
T *GetObjectSafe(TFile *f, const std::string &n, const std::string &m)
{
   auto v = reinterpret_cast<T *>(f->Get(m.c_str()));
   if (v == nullptr)
      throw std::runtime_error("Failed to read " + m + " from file " + n + ".");
   return v;
}
} // namespace Internal

/// Forest base class
///
/// \tparam T Value type for the computation (usually floating point type)
/// \tparam ForestType Type of the collection of trees
template <typename T, typename ForestType>
struct ForestBase {
   using Value_t = T;
   // TODO: Do we need the objective name?
   std::function<T(T)> fObjectiveFunc; ///< Objective function
   ForestType fTrees;                  ///< Store the forest, either as vector or jitted function
   int fNumFeatures;                   ///< Number of input variables / input features

   void Inference(const T *inputs, const int rows, T *predictions);
};

/// Perform inference of the forest on a batch of inputs
///
/// \param[in] inputs Pointer to data containing the inputs
/// \param[in] rows Number of events in inputs vector
/// \param[in] predictions Pointer to the buffer to be filled with the predictions
template <typename T, typename ForestType>
inline void ForestBase<T, ForestType>::Inference(const T *inputs, const int rows, T *predictions)
{
   for (int i = 0; i < rows; i++) {
      predictions[i] = 0.0;
      for (auto &tree : fTrees) {
         predictions[i] += tree.Inference(inputs + i * fNumFeatures);
      }
      predictions[i] = fObjectiveFunc(predictions[i]);
   }
}

/// Forest using branchless trees
///
/// \tparam T Value type for the computation (usually floating point type)
template <typename T>
struct BranchlessForest : public ForestBase<T, std::vector<BranchlessTree<T>>> {
   void Load(const std::string &key, const std::string &filename, bool sortTrees = true);
};

/// Load parameters from a ROOT file to the branchless trees
///
/// \param[in] key Name of folder in the ROOT file containing the model parameters
/// \param[in] filename Filename of the ROOT file
/// \param[in] sortTrees Flag to indicate sorting the input trees by the cut value of the first node of each tree
template <typename T>
inline void BranchlessForest<T>::Load(const std::string &key, const std::string &filename, bool sortTrees)
{
   // Open input file and get folder from key
   auto file = TFile::Open(filename.c_str(), "READ");

   // Load parameters from file
   // TODO: Can we sanitize loading the cut values of type X to the inference object of type Y?
   auto maxDepth = Internal::GetObjectSafe<std::vector<int>>(file, filename, key + "/max_depth");
   auto numTrees = Internal::GetObjectSafe<std::vector<int>>(file, filename, key + "/num_trees");
   auto numFeatures = Internal::GetObjectSafe<std::vector<int>>(file, filename, key + "/num_features");
   auto objective = Internal::GetObjectSafe<std::string>(file, filename, key + "/objective");
   auto features = Internal::GetObjectSafe<std::vector<int>>(file, filename, key + "/features");
   auto thresholds = Internal::GetObjectSafe<std::vector<T>>(file, filename, key + "/thresholds");

   this->fNumFeatures = numFeatures->at(0);
   this->fObjectiveFunc = Objectives::GetFunction<T>(*objective);
   this->fTrees.resize(numTrees->at(0));
   const auto lenFeatures = std::pow(2, maxDepth->at(0)) - 1;
   const auto lenThresholds = std::pow(2, maxDepth->at(0) + 1) - 1;

   // Sort trees by threshold value of first node
   // TODO

   // Load parameters in trees
   for (int i = 0; i < numTrees->at(0); i++) {
      // Set tree depth
      this->fTrees[i].fTreeDepth = maxDepth->at(0);

      // Set feature indices
      this->fTrees[i].fFeatures.resize(lenFeatures);
      for (int j = 0; j < lenFeatures; j++)
         this->fTrees[i].fFeatures[j] = features->at(i * lenFeatures + j);

      // Set threshold values
      this->fTrees[i].fThresholds.resize(lenThresholds);
      for (int j = 0; j < lenThresholds; j++)
         this->fTrees[i].fThresholds[j] = thresholds->at(i * lenThresholds + j);

      // Fill sparse trees fully
      this->fTrees[i].FillSparse();
   }

   // Clean-up
   delete maxDepth;
   delete numTrees;
   delete numFeatures;
   delete objective;
   delete features;
   delete thresholds;
   file->Close();
}

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_TREEINFERENCE_FOREST
