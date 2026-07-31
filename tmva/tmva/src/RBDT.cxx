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

#include <nlohmann/json.hpp>

#include <cmath>
#include <memory>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>

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

/// Construct an RBDT from an XGBoost model in its native JSON serialization.
///
/// In contrast to LoadText(), which parses the human-readable text dump, this
/// reads the structured model that XGBoost writes with Booster.save_model().
/// That format stores each tree as a set of parallel arrays and references
/// features by index, so no feature-name resolution is needed. Everything else
/// (objective, base score, number of classes) is taken from the file, which
/// makes this a self-contained, Python-free entry point.
TMVA::Experimental::RBDT TMVA::Experimental::RBDT::LoadXGBoost(std::string const &jsonPath)
{
   const std::string info = "constructing RBDT from '" + jsonPath + "': ";

   if (gSystem->AccessPathName(jsonPath.c_str())) {
      throw std::runtime_error(info + "file does not exist");
   }

   nlohmann::json j;
   {
      std::ifstream jsonFile(jsonPath.c_str());
      jsonFile >> j;
   }

   auto const &learner = j.at("learner");
   auto const &modelParam = learner.at("learner_model_param");

   // Map the XGBoost objective to the RBDT one, matching the Python SaveXGBoost.
   std::string const xgbObjective = learner.at("objective").at("name").get<std::string>();
   static const std::unordered_map<std::string, std::string> objectiveMap{
      {"multi:softprob", "softmax"}, // Naming the objective softmax is more common today
      {"binary:logistic", "logistic"},
      {"reg:linear", "identity"},
      {"reg:squarederror", "identity"},
   };
   auto foundObjective = objectiveMap.find(xgbObjective);
   if (foundObjective == objectiveMap.end()) {
      std::string supported;
      for (auto const &item : objectiveMap) {
         supported += (supported.empty() ? "" : ", ") + item.first;
      }
      throw std::runtime_error(info + "XGBoost model has unsupported objective \"" + xgbObjective +
                               "\". Supported objectives are " + supported + ".");
   }
   bool const logistic = foundObjective->second == "logistic";

   // The base score is stored as a string, e.g. "5.14E-1". Since XGBoost 3.1.0 it
   // is always serialized as a JSON array embedded in that string (e.g.
   // "[5.14E-1]"), even for single-output models. Only a genuine multi-element
   // array (multi-target base score) is unsupported.
   std::string const baseScoreStr = modelParam.at("base_score").get<std::string>();
   double baseScoreProb;
   if (baseScoreStr.find('[') != std::string::npos) {
      nlohmann::json const baseScoreArr = nlohmann::json::parse(baseScoreStr);
      if (baseScoreArr.size() > 1) {
         throw std::runtime_error(info + "model contains multiple base scores, which is not supported. This "
                                         "typically occurs with XGBoost >= 3.1.0, which supports multi-target base "
                                         "scores.");
      }
      baseScoreProb = baseScoreArr.at(0).get<double>();
   } else {
      baseScoreProb = std::stod(baseScoreStr);
   }
   // For a logistic objective the base score is a probability, but RBDT works on
   // the raw margin, so we apply the logit transform (as the Python code does).
   Value_t const baseScore = logistic ? std::log(baseScoreProb / (1.0 - baseScoreProb)) : baseScoreProb;

   // Only multiclass models produce more than one output.
   int nClasses = 1;
   if (xgbObjective.rfind("multi:", 0) == 0) {
      nClasses = std::stoi(modelParam.at("num_class").get<std::string>());
   }

   RBDT ff;
   ff.fLogistic = logistic;
   ff.fBaseScore = baseScore;
   ff.fBaseResponses.resize(nClasses <= 2 ? 1 : nClasses);

   auto const &trees = learner.at("gradient_booster").at("model").at("trees");

   int treesSkipped = 0;
   int nPreviousNodes = 0;
   int nPreviousLeaves = 0;
   IndexMap nodeIndices;
   IndexMap leafIndices;

   // Fill the flat RBDT arrays tree by tree, keying the index maps by the node's
   // position in the XGBoost arrays. terminateTree() then remaps the child
   // references to the RBDT indexing (negated for leaves), exactly as for the
   // text dump. Node 0 is always the tree root, so iterating in array order
   // makes it the first internal node of the tree, which is what fRootIndices
   // expects.
   for (auto const &tree : trees) {
      auto const &leftChildren = tree.at("left_children");
      auto const &rightChildren = tree.at("right_children");
      auto const &splitIndices = tree.at("split_indices");
      auto const &splitConditions = tree.at("split_conditions");

      std::size_t const nNodes = leftChildren.size();
      for (std::size_t i = 0; i < nNodes; ++i) {
         int const left = leftChildren[i].get<int>();
         if (left == -1) {
            // Leaf node: the split condition holds the leaf response.
            ff.fResponses.push_back(splitConditions[i].get<Value_t>());
            std::size_t const nLeafIndices = leafIndices.size();
            leafIndices[i] = nLeafIndices + nPreviousLeaves;
         } else {
            // Internal node: x < cut goes left (yes), otherwise right (no).
            ff.fCutValues.push_back(splitConditions[i].get<Value_t>());
            ff.fCutIndices.push_back(splitIndices[i].get<unsigned int>());
            ff.fLeftIndices.push_back(left);
            ff.fRightIndices.push_back(rightChildren[i].get<int>());
            std::size_t const nNodeIndices = nodeIndices.size();
            nodeIndices[i] = nNodeIndices + nPreviousNodes;
         }
      }

      terminateTree(ff, nPreviousNodes, nPreviousLeaves, nodeIndices, leafIndices, treesSkipped);
   }

   if (nClasses > 2 && (ff.fRootIndices.size() + treesSkipped) % nClasses != 0) {
      std::stringstream ss;
      ss << info << "Forest has " << ff.fRootIndices.size() << " trees, which is not compatible with " << nClasses
         << " classes!";
      throw std::runtime_error(ss.str());
   }

   return ff;
}

/// Save an XGBoost model to a ROOT file as a TMVA::Experimental::RBDT object.
///
/// \param jsonPath   Path to the XGBoost model in its native JSON serialization
///                   (as written by xgboost's Booster.save_model()).
/// \param keyName    Name under which the RBDT is stored in the output file.
/// \param outputPath Path of the ROOT file to create (opened in RECREATE mode).
///
/// This is the language-agnostic entry point for the XGBoost-to-ROOT path: it
/// only needs the model file on disk, so it can be used from C++ as well as
/// from Python.
void TMVA::Experimental::SaveXGBoost(std::string const &jsonPath, std::string const &keyName,
                                     std::string const &outputPath)
{
   RBDT bdt = RBDT::LoadXGBoost(jsonPath);

   std::unique_ptr<TFile> file{TFile::Open(outputPath.c_str(), "RECREATE")};
   if (!file || file->IsZombie()) {
      throw std::runtime_error("Failed to open output file " + outputPath);
   }
   file->WriteObject(&bdt, keyName.c_str());
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
