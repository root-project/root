#ifndef TMVA_RLEGACYREADER
#define TMVA_RLEGACYREADER

#include "ROOT/RVec.hxx"
#include "TMVA/RTensor.hxx"
#include "TMVA/Reader.h"
#include "TString.h"
#include "TXMLEngine.h"
#include <memory>

namespace TMVA {
namespace Experimental {

namespace Internal {

/// Internal definition of analysis types
enum AnalysisType : unsigned int { Undefined = 0, Classification, Regression, Multiclass };

/// Container for information extracted from TMVA XML config
struct LegacyConfig {
   unsigned int numVariables;
   std::vector<std::string> variables;
   unsigned int numClasses;
   std::vector<std::string> classes;
   AnalysisType analysisType;
   LegacyConfig()
      : numVariables(0), variables(std::vector<std::string>(0)), numClasses(0), classes(std::vector<std::string>(0)),
        analysisType(Internal::AnalysisType::Undefined)
   {
   }
};

/// Parse TMVA XML config
inline LegacyConfig ParseLegacyConfig(const std::string &filename)
{
   LegacyConfig c;

   // Parse XML file and find root node
   TXMLEngine xml;
   auto xmldoc = xml.ParseFile(filename.c_str());
   auto mainNode = xml.DocGetRootElement(xmldoc);
   for (auto node = xml.GetChild(mainNode); node; node = xml.GetNext(node)) {
      const auto nodeName = std::string(xml.GetNodeName(node));
      // Read out input variables
      if (nodeName.compare("Variables") == 0) {
         c.numVariables = std::atoi(xml.GetAttr(node, "NVar"));
         c.variables = std::vector<std::string>(c.numVariables);
         for (auto thisNode = xml.GetChild(node); thisNode; thisNode = xml.GetNext(thisNode)) {
            const auto iVariable = std::atoi(xml.GetAttr(thisNode, "VarIndex"));
            c.variables[iVariable] = xml.GetAttr(thisNode, "Title");
         }
      }
      // Read out output classes
      else if (nodeName.compare("Classes") == 0) {
         c.numClasses = std::atoi(xml.GetAttr(node, "NClass"));
         for (auto thisNode = xml.GetChild(node); thisNode; thisNode = xml.GetNext(thisNode)) {
            c.classes.push_back(xml.GetAttr(thisNode, "Name"));
         }
      }
      // Read out analysis type
      else if (nodeName.compare("GeneralInfo") == 0) {
         std::string analysisType = "";
         for (auto thisNode = xml.GetChild(node); thisNode; thisNode = xml.GetNext(thisNode)) {
            if (std::string("AnalysisType").compare(xml.GetAttr(thisNode, "name")) == 0) {
               analysisType = xml.GetAttr(thisNode, "value");
            }
         }
         if (analysisType.compare("Classification") == 0) {
            c.analysisType = Internal::AnalysisType::Classification;
         } else if (analysisType.compare("Regression") == 0) {
            c.analysisType = Internal::AnalysisType::Regression;
         } else if (analysisType.compare("MultiClass") == 0) {
            c.analysisType = Internal::AnalysisType::Multiclass;
         }
      }
   }
   xml.FreeDoc(xmldoc);

   // Error-handling
   if (c.numVariables != c.variables.size() || c.numVariables == 0) {
      std::stringstream ss;
      ss << "Failed to parse input variables from TMVA config " << filename << ".";
      throw std::runtime_error(ss.str());
   }
   if (c.numClasses != c.classes.size() || c.numClasses == 0) {
      std::stringstream ss;
      ss << "Failed to parse output classes from TMVA config " << filename << ".";
      throw std::runtime_error(ss.str());
   }
   if (c.analysisType == Internal::AnalysisType::Undefined) {
      std::stringstream ss;
      ss << "Failed to parse analysis type from TMVA config " << filename << ".";
      throw std::runtime_error(ss.str());
   }

   return c;
}

/// Predict helper
template <typename I, typename T, typename F>
class PredictHelper;

template <std::size_t... N, typename T, typename F>
class PredictHelper<std::index_sequence<N...>, T, F> {
   template <std::size_t Idx>
   using AlwaysT = T;
   F fFunc;

public:
   PredictHelper(F &&f) : fFunc(std::forward<F>(f)) {}
   auto operator()(AlwaysT<N>... args) -> decltype(fFunc.Predict({args...})) { return fFunc.Predict({args...}); }
};

} // namespace Internal

/// TMVA::Reader legacy interface
class RLegacyReader {
   std::unique_ptr<Reader> fReader;
   std::vector<float> fValues;
   std::vector<std::string> fVariables;
   const char *name = "RLegacyReader";
   Internal::AnalysisType fAnalysisType;

public:
   RLegacyReader(const std::string &path)
   {
      // Load config
      auto c = Internal::ParseLegacyConfig(path);
      fVariables = c.variables;
      fAnalysisType = c.analysisType;

      // Setup reader
      fReader = std::make_unique<Reader>("Silent");
      const auto numVars = fVariables.size();
      fValues = std::vector<float>(numVars);
      for (unsigned int i = 0; i < numVars; i++) {
         fReader->AddVariable(TString(fVariables[i]), &fValues[i]);
      }
      fReader->BookMVA(name, path.c_str());
   }

   std::vector<float> Predict(const std::vector<float> &x)
   {
      for (std::size_t i = 0; i < x.size(); i++) {
         fValues[i] = x[i];
      }
      R__WRITE_LOCKGUARD(ROOT::gCoreMutex);
      if (fAnalysisType == Internal::AnalysisType::Classification) {
         return std::vector<float>({static_cast<float>(fReader->EvaluateMVA(name))});
      } else if (fAnalysisType == Internal::AnalysisType::Regression) {
         std::vector<float> prediction;
         for (const auto &p : fReader->EvaluateRegression(name)) {
            prediction.push_back(static_cast<float>(p));
         }
         return prediction;
      } else if (fAnalysisType == Internal::AnalysisType::Multiclass) {
         std::vector<float> prediction;
         for (const auto &p : fReader->EvaluateMulticlass(name)) {
            prediction.push_back(static_cast<float>(p));
         }
         return prediction;
      } else {
         throw std::runtime_error("RLegacyReader has undefined analysis type.");
         return std::vector<float>();
      }
   }

   RTensor<float> Predict(RTensor<float> &x)
   {
      const auto shape = x.GetShape();
      const auto numEntries = shape[0];
      const auto numVars = shape[1];
      RTensor<float> y({numEntries});
      for (std::size_t i = 0; i < numEntries; i++) {
         for (std::size_t j = 0; j < numVars; j++) {
            fValues[j] = x(i, j);
         }
         R__WRITE_LOCKGUARD(ROOT::gCoreMutex);
         y(i) = fReader->EvaluateMVA(name);
      }
      return y;
   }

   std::vector<std::string> GetVariableNames() { return fVariables; }
};

/// Helper to pass TMVA model to RDataFrame.Define nodes
template <std::size_t N, typename T, typename F>
auto Predict(F &&f) -> Internal::PredictHelper<std::make_index_sequence<N>, T, F>
{
   return Internal::PredictHelper<std::make_index_sequence<N>, T, F>(std::forward<F>(f));
}

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RLEGACYREADER
