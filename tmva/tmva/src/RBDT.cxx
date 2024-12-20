/**********************************************************************************
 * Project: ROOT - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 *                                                                                *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Authors:                                                                       *
 *      Jonas Rembser (jonas.rembser@cern.ch)                                     *
 *                                                                                *
 * Copyright (c) 2024:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#include <TMVA/RBDT.hxx>

#include <ROOT/StringUtils.hxx>

#include <TFile.h>
#include <TSystem.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>

namespace {

template <class Value_t>
void softmaxTransformInplace(Value_t *out, int nOut)
{
   // Do softmax transformation inplace, mimicing exactly the Softmax function
   // in the src/common/math.h source file of xgboost.
   double norm = 0.;
   Value_t wmax = *out;
   for (int i = 1; i < nOut; ++i) {
      wmax = std::max(out[i], wmax);
   }
   for (int i = 0; i < nOut; ++i) {
      Value_t &x = out[i];
      x = std::exp(x - wmax);
      norm += x;
   }
   for (int i = 0; i < nOut; ++i) {
      out[i] /= static_cast<float>(norm);
   }
}

namespace util {

inline bool isInteger(const std::string &s)
{
   if (s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+')))
      return false;

   char *p;
   strtol(s.c_str(), &p, 10);

   return (*p == 0);
}

template <class NumericType>
struct NumericAfterSubstrOutput {
   explicit NumericAfterSubstrOutput()
   {
      value = 0;
      found = false;
      failed = true;
   }
   NumericType value;
   bool found;
   bool failed;
   std::string rest;
};

template <class NumericType>
inline NumericAfterSubstrOutput<NumericType> numericAfterSubstr(std::string const &str, std::string const &substr)
{
   std::string rest;
   NumericAfterSubstrOutput<NumericType> output;
   output.rest = str;

   std::size_t found = str.find(substr);
   if (found != std::string::npos) {
      output.found = true;
      std::stringstream ss(str.substr(found + substr.size(), str.size() - found + substr.size()));
      ss >> output.value;
      if (!ss.fail()) {
         output.failed = false;
         output.rest = ss.str();
      }
   }
   return output;
}

} // namespace util

} // namespace

using TMVA::Experimental::RTensor;

/// Compute model prediction on input RTensor
RTensor<TMVA::Experimental::RBDT::Value_t> TMVA::Experimental::RBDT::Compute(RTensor<Value_t> const &x) const
{
   std::size_t nOut = fBaseResponses.size() > 2 ? fBaseResponses.size() : 1;
   const std::size_t rows = x.GetShape()[0];
   const std::size_t cols = x.GetShape()[1];
   RTensor<Value_t> y({rows, nOut}, MemoryLayout::ColumnMajor);
   std::vector<Value_t> xRow(cols);
   std::vector<Value_t> yRow(nOut);
   for (std::size_t iRow = 0; iRow < rows; ++iRow) {
      for (std::size_t iCol = 0; iCol < cols; ++iCol) {
         xRow[iCol] = x({iRow, iCol});
      }
      ComputeImpl(xRow.data(), yRow.data());
      for (std::size_t iOut = 0; iOut < nOut; ++iOut) {
         y({iRow, iOut}) = yRow[iOut];
      }
   }
   return y;
}

void TMVA::Experimental::RBDT::Softmax(const Value_t *array, Value_t *out) const
{
   std::size_t nOut = fBaseResponses.size() > 2 ? fBaseResponses.size() : 1;
   if (nOut == 1) {
      throw std::runtime_error(
         "Error in RBDT::softmax : binary classification models don't support softmax evaluation. Plase set "
         "the number of classes in the RBDT-creating function if this is a multiclassification model.");
   }

   for (std::size_t i = 0; i < nOut; ++i) {
      out[i] = fBaseScore + fBaseResponses[i];
   }

   int iRootIndex = 0;
   for (int index : fRootIndices) {
      do {
         int r = fRightIndices[index];
         int l = fLeftIndices[index];
         index = array[fCutIndices[index]] < fCutValues[index] ? l : r;
      } while (index > 0);
      out[fTreeNumbers[iRootIndex] % nOut] += fResponses[-index];
      ++iRootIndex;
   }

   softmaxTransformInplace(out, nOut);
}

void TMVA::Experimental::RBDT::ComputeImpl(const Value_t *array, Value_t *out) const
{
   std::size_t nOut = fBaseResponses.size() > 2 ? fBaseResponses.size() : 1;
   if (nOut > 1) {
      Softmax(array, out);
   } else {
      out[0] = EvaluateBinary(array);
      if (fLogistic) {
         out[0] = 1.0 / (1.0 + std::exp(-out[0]));
      }
   }
}

TMVA::Experimental::RBDT::Value_t TMVA::Experimental::RBDT::EvaluateBinary(const Value_t *array) const
{
   Value_t out = fBaseScore + fBaseResponses[0];

   for (std::vector<int>::const_iterator indexIter = fRootIndices.begin(); indexIter != fRootIndices.end();
        ++indexIter) {
      int index = *indexIter;
      do {
         int r = fRightIndices[index];
         int l = fLeftIndices[index];
         index = array[fCutIndices[index]] < fCutValues[index] ? l : r;
      } while (index > 0);
      out += fResponses[-index];
   }

   return out;
}

/// RBDT uses a more efficient representation of the BDT in flat arrays. This
/// function translates the indices to the RBDT indices. In RBDT, leaf nodes
/// are stored in separate arrays. To encode this, the sign of the index is
/// flipped.
void TMVA::Experimental::RBDT::correctIndices(std::span<int> indices, IndexMap const &nodeIndices,
                                              IndexMap const &leafIndices)
{
   for (int &idx : indices) {
      auto foundNode = nodeIndices.find(idx);
      if (foundNode != nodeIndices.end()) {
         idx = foundNode->second;
         continue;
      }
      auto foundLeaf = leafIndices.find(idx);
      if (foundLeaf != leafIndices.end()) {
         idx = -foundLeaf->second;
         continue;
      } else {
         std::stringstream errMsg;
         errMsg << "RBDT: something is wrong in the node structure - node with index " << idx << " doesn't exist";
         throw std::runtime_error(errMsg.str());
      }
   }
}

void TMVA::Experimental::RBDT::terminateTree(TMVA::Experimental::RBDT &ff, int &nPreviousNodes, int &nPreviousLeaves,
                                             IndexMap &nodeIndices, IndexMap &leafIndices, int &treesSkipped)
{
   correctIndices({ff.fRightIndices.begin() + nPreviousNodes, ff.fRightIndices.end()}, nodeIndices, leafIndices);
   correctIndices({ff.fLeftIndices.begin() + nPreviousNodes, ff.fLeftIndices.end()}, nodeIndices, leafIndices);

   if (nPreviousNodes != static_cast<int>(ff.fCutValues.size())) {
      ff.fTreeNumbers.push_back(ff.fRootIndices.size() + treesSkipped);
      ff.fRootIndices.push_back(nPreviousNodes);
   } else {
      int treeNumbers = ff.fRootIndices.size() + treesSkipped;
      ++treesSkipped;
      ff.fBaseResponses[treeNumbers % ff.fBaseResponses.size()] += ff.fResponses.back();
      ff.fResponses.pop_back();
   }

   nodeIndices.clear();
   leafIndices.clear();
   nPreviousNodes = ff.fCutValues.size();
   nPreviousLeaves = ff.fResponses.size();
}

TMVA::Experimental::RBDT TMVA::Experimental::RBDT::LoadText(std::string const &txtpath,
                                                            std::vector<std::string> &features, int nClasses,
                                                            bool logistic, Value_t baseScore)
{
   const std::string info = "constructing RBDT from " + txtpath + ": ";

   if (gSystem->AccessPathName(txtpath.c_str())) {
      throw std::runtime_error(info + "file does not exists");
   }

   std::ifstream file(txtpath.c_str());
   return LoadText(file, features, nClasses, logistic, baseScore);
}

TMVA::Experimental::RBDT TMVA::Experimental::RBDT::LoadText(std::istream &file, std::vector<std::string> &features,
                                                            int nClasses, bool logistic, Value_t baseScore)
{
   const std::string info = "constructing RBDT from istream: ";

   RBDT ff;
   ff.fLogistic = logistic;
   ff.fBaseScore = baseScore;
   ff.fBaseResponses.resize(nClasses <= 2 ? 1 : nClasses);

   int treesSkipped = 0;

   int nVariables = 0;
   std::unordered_map<std::string, int> varIndices;
   bool fixFeatures = false;

   if (!features.empty()) {
      fixFeatures = true;
      nVariables = features.size();
      for (int i = 0; i < nVariables; ++i) {
         varIndices[features[i]] = i;
      }
   }

   std::string line;

   IndexMap nodeIndices;
   IndexMap leafIndices;

   int nPreviousNodes = 0;
   int nPreviousLeaves = 0;

   while (std::getline(file, line)) {
      std::size_t foundBegin = line.find("[");
      std::size_t foundEnd = line.find("]");
      if (foundBegin != std::string::npos) {
         std::string subline = line.substr(foundBegin + 1, foundEnd - foundBegin - 1);
         if (util::isInteger(subline) && !ff.fResponses.empty()) {
            terminateTree(ff, nPreviousNodes, nPreviousLeaves, nodeIndices, leafIndices, treesSkipped);
         } else if (!util::isInteger(subline)) {
            std::stringstream ss(line);
            int index;
            ss >> index;
            line = ss.str();

            std::vector<std::string> splitstring = ROOT::Split(subline, "<");
            std::string const &varName = splitstring[0];
            Value_t cutValue;
            {
               std::stringstream ss1(splitstring[1]);
               ss1 >> cutValue;
            }
            if (!varIndices.count(varName)) {
               if (fixFeatures) {
                  throw std::runtime_error(info + "feature " + varName + " not in list of features");
               }
               varIndices[varName] = nVariables;
               features.push_back(varName);
               ++nVariables;
            }
            int yes;
            int no;
            util::NumericAfterSubstrOutput<int> output = util::numericAfterSubstr<int>(line, "yes=");
            if (!output.failed) {
               yes = output.value;
            } else {
               throw std::runtime_error(info + "problem while parsing the text dump");
            }
            output = util::numericAfterSubstr<int>(output.rest, "no=");
            if (!output.failed) {
               no = output.value;
            } else {
               throw std::runtime_error(info + "problem while parsing the text dump");
            }

            ff.fCutValues.push_back(cutValue);
            ff.fCutIndices.push_back(varIndices[varName]);
            ff.fLeftIndices.push_back(yes);
            ff.fRightIndices.push_back(no);
            std::size_t nNodeIndices = nodeIndices.size();
            nodeIndices[index] = nNodeIndices + nPreviousNodes;
         }

      } else {
         util::NumericAfterSubstrOutput<Value_t> output = util::numericAfterSubstr<Value_t>(line, "leaf=");
         if (output.found) {
            std::stringstream ss(line);
            int index;
            ss >> index;
            line = ss.str();

            ff.fResponses.push_back(output.value);
            std::size_t nLeafIndices = leafIndices.size();
            leafIndices[index] = nLeafIndices + nPreviousLeaves;
         }
      }
   }
   terminateTree(ff, nPreviousNodes, nPreviousLeaves, nodeIndices, leafIndices, treesSkipped);

   if (nClasses > 2 && (ff.fRootIndices.size() + treesSkipped) % nClasses != 0) {
      std::stringstream ss;
      ss << "Error in RBDT construction : Forest has " << ff.fRootIndices.size()
         << " trees, which is not compatible with " << nClasses << "classes!";
      throw std::runtime_error(ss.str());
   }

   return ff;
}

TMVA::Experimental::RBDT::RBDT(const std::string &key, const std::string &filename)
{
   std::unique_ptr<TFile> file{TFile::Open(filename.c_str(), "READ")};
   if (!file || file->IsZombie()) {
      throw std::runtime_error("Failed to open input file " + filename);
   }
   auto *fromFile = file->Get<TMVA::Experimental::RBDT>(key.c_str());
   if (!fromFile) {
      throw std::runtime_error("No RBDT with name " + key);
   }
   *this = *fromFile;
}
