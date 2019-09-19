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

template <typename T>
bool CompareTree(const BranchlessTree<T> &a, const BranchlessTree<T> &b)
{
   if (a.fInputs[0] == b.fInputs[0])
      return a.fThresholds[0] < b.fThresholds[0];
   else
      return a.fInputs[0] < b.fInputs[0];
}
} // namespace Internal

/// Forest base class
///
/// \tparam T Value type for the computation (usually floating point type)
/// \tparam ForestType Type of the collection of trees
template <typename T, typename ForestType>
struct ForestBase {
   using Value_t = T;
   std::function<T(T)> fObjectiveFunc; ///< Objective function
   ForestType fTrees;                  ///< Store the forest, either as vector or jitted function
   int fNumInputs;                     ///< Number of input variables

   void Inference(const T *inputs, const int rows, bool layout, T *predictions);
};

/// Perform inference of the forest on a batch of inputs
///
/// \param[in] inputs Pointer to data containing the inputs
/// \param[in] rows Number of events in inputs vector
/// \param[in] layout Row major (true) or column major (false) memory layout
/// \param[in] predictions Pointer to the buffer to be filled with the predictions
template <typename T, typename ForestType>
inline void ForestBase<T, ForestType>::Inference(const T *inputs, const int rows, bool layout, T *predictions)
{
   const auto strideTree = layout ? 1 : rows;
   const auto strideBatch = layout ? fNumInputs : 1;
   for (int i = 0; i < rows; i++) {
      predictions[i] = 0.0;
      for (auto &tree : fTrees) {
         predictions[i] += tree.Inference(inputs + i * strideBatch, strideTree);
      }
      predictions[i] = fObjectiveFunc(predictions[i]);
   }
}

/// Forest using branchless trees
///
/// \tparam T Value type for the computation (usually floating point type)
template <typename T>
struct BranchlessForest : public ForestBase<T, std::vector<BranchlessTree<T>>> {
   void Load(const std::string &key, const std::string &filename, const int output = 0, const bool sortTrees = true);
};

/// Load parameters from a ROOT file to the branchless trees
///
/// \param[in] key Name of folder in the ROOT file containing the model parameters
/// \param[in] filename Filename of the ROOT file
/// \param[in] output Load trees corresponding to the given output node of the forest
/// \param[in] sortTrees Flag to indicate sorting the input trees by the cut value of the first node of each tree
template <typename T>
inline void
BranchlessForest<T>::Load(const std::string &key, const std::string &filename, const int output, const bool sortTrees)
{
   // Open input file and get folder from key
   auto file = TFile::Open(filename.c_str(), "READ");

   // Load parameters from file
   auto maxDepth = Internal::GetObjectSafe<std::vector<int>>(file, filename, key + "/max_depth");
   auto numTrees = Internal::GetObjectSafe<std::vector<int>>(file, filename, key + "/num_trees");
   auto numInputs = Internal::GetObjectSafe<std::vector<int>>(file, filename, key + "/num_inputs");
   auto numOutputs = Internal::GetObjectSafe<std::vector<int>>(file, filename, key + "/num_outputs");
   auto objective = Internal::GetObjectSafe<std::string>(file, filename, key + "/objective");
   auto inputs = Internal::GetObjectSafe<std::vector<int>>(file, filename, key + "/inputs");
   auto outputs = Internal::GetObjectSafe<std::vector<int>>(file, filename, key + "/outputs");
   auto thresholds = Internal::GetObjectSafe<std::vector<T>>(file, filename, key + "/thresholds");

   this->fNumInputs = numInputs->at(0);
   this->fObjectiveFunc = Objectives::GetFunction<T>(*objective);
   const auto lenInputs = std::pow(2, maxDepth->at(0)) - 1;
   const auto lenThresholds = std::pow(2, maxDepth->at(0) + 1) - 1;

   // Find number of trees corresponding to given output node
   if (output > numOutputs->at(0))
      throw std::runtime_error("Given output node of the forest is larger or equal to number of output nodes.");
   int c = 0;
   for (int i = 0; i < numTrees->at(0); i++)
      if (outputs->at(i) == output)
         c++;
   if (c == 0)
      std::runtime_error("No trees found for given output node of the forest.");
   this->fTrees.resize(c);

   // Load parameters in trees
   c = 0;
   for (int i = 0; i < numTrees->at(0); i++) {
      // Select only trees for the given output node of the forest
      if (outputs->at(i) != output)
         continue;

      // Set tree depth
      this->fTrees[c].fTreeDepth = maxDepth->at(0);

      // Set feature indices
      this->fTrees[c].fInputs.resize(lenInputs);
      for (int j = 0; j < lenInputs; j++)
         this->fTrees[c].fInputs[j] = inputs->at(i * lenInputs + j);

      // Set threshold values
      this->fTrees[c].fThresholds.resize(lenThresholds);
      for (int j = 0; j < lenThresholds; j++)
         this->fTrees[c].fThresholds[j] = thresholds->at(i * lenThresholds + j);

      // Fill sparse trees fully
      this->fTrees[c].FillSparse();

      c++;
   }

   // Sort trees by first cut variable and threshold value
   if (sortTrees)
      std::sort(this->fTrees.begin(), this->fTrees.end(), Internal::CompareTree<T>);

   // Clean-up
   delete maxDepth;
   delete numTrees;
   delete numInputs;
   delete objective;
   delete inputs;
   delete thresholds;
   file->Close();
}

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_TREEINFERENCE_FOREST
