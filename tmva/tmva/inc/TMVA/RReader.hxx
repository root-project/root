#ifndef TMVA_RREADER
#define TMVA_RREADER

#include "TString.h"
#include "TXMLEngine.h"

#include "TMVA/RTensor.hxx"
#include "TMVA/Reader.h"

#include <memory> // std::unique_ptr
#include <sstream> // std::stringstream

namespace TMVA {
namespace Experimental {

namespace Internal {

/// Internal definition of analysis types
enum AnalysisType : unsigned int { Undefined = 0, Classification, Regression, Multiclass };

/// Container for information extracted from TMVA XML config
struct XMLConfig {
   unsigned int numVariables;
   std::vector<std::string> variables;
   std::vector<std::string> expressions;
   unsigned int numClasses;
   std::vector<std::string> classes;
   AnalysisType analysisType;
   XMLConfig()
      : numVariables(0), variables(std::vector<std::string>(0)), numClasses(0), classes(std::vector<std::string>(0)),
        analysisType(Internal::AnalysisType::Undefined)
   {
   }
};

/// Parse TMVA XML config
inline XMLConfig ParseXMLConfig(const std::string &filename)
{
   XMLConfig c;

   // Parse XML file and find root node
   TXMLEngine xml;
   auto xmldoc = xml.ParseFile(filename.c_str());
   if (!xmldoc) {
      std::stringstream ss;
      ss << "Failed to open TMVA XML file "
         << filename << ".";
      throw std::runtime_error(ss.str());
   }
   auto mainNode = xml.DocGetRootElement(xmldoc);
   for (auto node = xml.GetChild(mainNode); node; node = xml.GetNext(node)) {
      const auto nodeName = std::string(xml.GetNodeName(node));
      // Read out input variables
      if (nodeName.compare("Variables") == 0) {
         c.numVariables = std::atoi(xml.GetAttr(node, "NVar"));
         c.variables = std::vector<std::string>(c.numVariables);
         c.expressions = std::vector<std::string>(c.numVariables);
         for (auto thisNode = xml.GetChild(node); thisNode; thisNode = xml.GetNext(thisNode)) {
            const auto iVariable = std::atoi(xml.GetAttr(thisNode, "VarIndex"));
            c.variables[iVariable] = xml.GetAttr(thisNode, "Title");
            c.expressions[iVariable] = xml.GetAttr(thisNode, "Expression");
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
         } else if (analysisType.compare("Multiclass") == 0) {
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

} // namespace Internal

/// TMVA::Reader legacy interface
class RReader {
private:
   std::unique_ptr<Reader> fReader;
   std::vector<float> fValues;
   std::vector<std::string> fVariables;
   std::vector<std::string> fExpressions;
   unsigned int fNumClasses;
   const char *name = "RReader";
   Internal::AnalysisType fAnalysisType;

public:
   /// Create TMVA model from XML file
   RReader(const std::string &path)
   {
      // Load config
      auto c = Internal::ParseXMLConfig(path);
      fVariables = c.variables;
      fExpressions = c.expressions;
      fAnalysisType = c.analysisType;
      fNumClasses = c.numClasses;

      // Setup reader
      fReader = std::make_unique<Reader>("Silent");
      const auto numVars = fVariables.size();
      fValues = std::vector<float>(numVars);
      for (std::size_t i = 0; i < numVars; i++) {
         fReader->AddVariable(TString(fExpressions[i]), &fValues[i]);
      }
      fReader->BookMVA(name, path.c_str());
   }

   /// Compute model prediction on vector
   std::vector<float> Compute(const std::vector<float> &x)
   {
      if (x.size() != fVariables.size())
         throw std::runtime_error("Size of input vector is not equal to number of variables.");

      // Copy over inputs to memory used by TMVA reader
      for (std::size_t i = 0; i < x.size(); i++) {
         fValues[i] = x[i];
      }

      // Take lock to protect model evaluation
      R__WRITE_LOCKGUARD(ROOT::gCoreMutex);

      // Evaluate TMVA model
      // Classification
      if (fAnalysisType == Internal::AnalysisType::Classification) {
         return std::vector<float>({static_cast<float>(fReader->EvaluateMVA(name))});
      }
      // Regression
      else if (fAnalysisType == Internal::AnalysisType::Regression) {
         return fReader->EvaluateRegression(name);
      }
      // Multiclass
      else if (fAnalysisType == Internal::AnalysisType::Multiclass) {
         return fReader->EvaluateMulticlass(name);
      }
      // Throw error
      else {
         throw std::runtime_error("RReader has undefined analysis type.");
         return std::vector<float>();
      }
   }

   /// Compute model prediction on input RTensor
   RTensor<float> Compute(RTensor<float> &x)
   {
      // Error-handling for input tensor
      const auto shape = x.GetShape();
      if (shape.size() != 2)
         throw std::runtime_error("Can only compute model outputs for input tensor of rank 2.");

      const auto numEntries = shape[0];
      const auto numVars = shape[1];
      if (numVars != fVariables.size())
         throw std::runtime_error("Second dimension of input tensor is not equal to number of variables.");

      // Define shape of output tensor based on analysis type
      unsigned int numClasses = 1;
      if (fAnalysisType == Internal::AnalysisType::Multiclass)
         numClasses = fNumClasses;
      RTensor<float> y({numEntries * numClasses});
      if (fAnalysisType == Internal::AnalysisType::Multiclass)
         y = y.Reshape({numEntries, numClasses});

      // Fill output tensor
      for (std::size_t i = 0; i < numEntries; i++) {
         for (std::size_t j = 0; j < numVars; j++) {
            fValues[j] = x(i, j);
         }
         R__WRITE_LOCKGUARD(ROOT::gCoreMutex);
         // Classification
         if (fAnalysisType == Internal::AnalysisType::Classification) {
            y(i) = fReader->EvaluateMVA(name);
         }
         // Regression
         else if (fAnalysisType == Internal::AnalysisType::Regression) {
            y(i) = fReader->EvaluateRegression(name)[0];
         }
         // Multiclass
         else if (fAnalysisType == Internal::AnalysisType::Multiclass) {
            const auto p = fReader->EvaluateMulticlass(name);
            for (std::size_t k = 0; k < numClasses; k++)
               y(i, k) = p[k];
         }
      }

      return y;
   }

   std::vector<std::string> GetVariableNames() { return fVariables; }
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RREADER
