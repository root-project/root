/**********************************************************************************
 * Project: ROOT - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *                                        *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Authors:                                                                       *
 *      Lorenzo Moneta                                  *
 *                                                                                *
 * Copyright (c) 2022:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 **********************************************************************************/


#ifndef TMVA_RSOFIEREADER
#define TMVA_RSOFIEREADER


#include <string>
#include <vector>
#include <memory> // std::unique_ptr
#include <sstream> // std::stringstream
#include <iostream>
#include "TROOT.h"
#include "TSystem.h"
#include "TError.h"
#include "TInterpreter.h"
#include "TUUID.h"
#include "TMVA/RTensor.hxx"
#include "Math/Util.h"

namespace TMVA {
namespace Experimental {




/// TMVA::RSofieReader class for reading external Machine Learning models
/// in ONNX files, Keras .h5 files or PyTorch .pt files
/// and performing the inference using SOFIE
/// It is reccomended to use ONNX if possible since there is a larger support for
/// model operators.

class RSofieReader  {


public:
   /// Dummy constructor which needs model loading  afterwards
   RSofieReader() {}
   /// Create TMVA model from ONNX file
   /// print level can be 0 (minimal) 1 with info , 2 with all ONNX parsing info
   RSofieReader(const std::string &path, std::vector<std::vector<size_t>> inputShapes = {}, int verbose = 0)
   {
      Load(path, inputShapes, verbose);
   }

   void Load(const std::string &path, std::vector<std::vector<size_t>> inputShapes = {}, int verbose = 0)
   {

      enum EModelType {kONNX, kKeras, kPt, kROOT, kNotDef}; // type of model
      EModelType type = kNotDef;

      auto pos1 = path.rfind("/");
      auto pos2 = path.find(".onnx");
      if (pos2 != std::string::npos) {
         type = kONNX;
      } else {
         pos2 = path.find(".h5");
         if (pos2 != std::string::npos) {
             type = kKeras;
         } else {
            pos2 = path.find(".pt");
            if (pos2 != std::string::npos) {
               type = kPt;
            }
            else {
               pos2 = path.find(".root");
               if (pos2 != std::string::npos) {
                  type = kROOT;
               }
            }
         }
      }
      if (type == kNotDef) {
         throw std::runtime_error("Input file is not an ONNX or Keras or PyTorch file");
      }
      if (pos1 == std::string::npos)
         pos1 = 0;
      else
         pos1 += 1;
      std::string modelName = path.substr(pos1,pos2-pos1);
      std::string fileType = path.substr(pos2+1, path.length()-pos2-1);
      if (verbose) std::cout << "Parsing SOFIE model " << modelName << " of type " << fileType << std::endl;

      // create code for parsing model and generate C++ code for inference
      // make it in a separate scope to avoid polluting global interpreter space
      std::string parserCode;
      if (type == kONNX) {
         // check first if we can load the SOFIE parser library
         if (gSystem->Load("libROOTTMVASofieParser") < 0) {
            throw std::runtime_error("RSofieReader: cannot use SOFIE with ONNX since libROOTTMVASofieParser is missing");
         }
         gInterpreter->Declare("#include \"TMVA/RModelParser_ONNX.hxx\"");
         parserCode += "{\nTMVA::Experimental::SOFIE::RModelParser_ONNX parser ; \n";
         if (verbose == 2)
            parserCode += "TMVA::Experimental::SOFIE::RModel model = parser.Parse(\"" + path + "\",true); \n";
         else
            parserCode += "TMVA::Experimental::SOFIE::RModel model = parser.Parse(\"" + path + "\"); \n";
      }
      else if (type == kKeras) {
         // use Keras direct parser
         if (gSystem->Load("libPyMVA") < 0) {
            throw std::runtime_error("RSofieReader: cannot use SOFIE with Keras since libPyMVA is missing");
         }
         // assume batch size is first entry in first input !
         std::string batch_size = "-1";
         if (!inputShapes.empty() && ! inputShapes[0].empty())
            batch_size = std::to_string(inputShapes[0][0]);
         parserCode += "{\nTMVA::Experimental::SOFIE::RModel model = TMVA::Experimental::SOFIE::PyKeras::Parse(\"" + path +
                       "\"," + batch_size + "); \n";
      }
      else if (type == kPt) {
         // use PyTorch direct parser
         if (gSystem->Load("libPyMVA") < 0) {
            throw std::runtime_error("RSofieReader: cannot use SOFIE with PyTorch since libPyMVA is missing");
         }
         if (inputShapes.size() == 0) {
            throw std::runtime_error("RSofieReader: cannot use SOFIE with PyTorch since the input tensor shape is missing and is needed by the PyTorch parser");
         }
         std::string inputShapesStr = "{";
         for (unsigned int i = 0; i < inputShapes.size(); i++) {
            inputShapesStr += "{ ";
            for (unsigned int j = 0; j < inputShapes[i].size(); j++) {
               inputShapesStr += ROOT::Math::Util::ToString(inputShapes[i][j]);
               if (j < inputShapes[i].size()-1) inputShapesStr += ", ";
            }
            inputShapesStr += "}";
            if (i < inputShapes.size()-1) inputShapesStr += ", ";
         }
         inputShapesStr += "}";
         parserCode += "{\nTMVA::Experimental::SOFIE::RModel model = TMVA::Experimental::SOFIE::PyTorch::Parse(\"" + path + "\", "
                    + inputShapesStr + "); \n";
      }
      else if (type == kROOT) {
         // use  parser from ROOT
         parserCode += "{\nauto fileRead = TFile::Open(\"" + path + "\",\"READ\");\n";
         parserCode += "TMVA::Experimental::SOFIE::RModel * modelPtr;\n";
         parserCode += "auto keyList = fileRead->GetListOfKeys(); TString name;\n";
         parserCode += "for (const auto&& k : *keyList)  { \n";
         parserCode += "   TString cname =  ((TKey*)k)->GetClassName();  if (cname==\"TMVA::Experimental::SOFIE::RModel\") name = k->GetName(); }\n";
         parserCode += "fileRead->GetObject(name,modelPtr); fileRead->Close(); delete fileRead;\n";
         parserCode += "TMVA::Experimental::SOFIE::RModel & model = *modelPtr;\n";
      }

       // add custom operators if needed
      if (fCustomOperators.size() > 0) {

         for (auto & op : fCustomOperators) {
            parserCode += "{ auto p = new TMVA::Experimental::SOFIE::ROperator_Custom<float>(\""
                      + op.fOpName + "\"," + op.fInputNames + "," + op.fOutputNames + "," + op.fOutputShapes + ",\"" + op.fFileName + "\");\n";
            parserCode += "std::unique_ptr<TMVA::Experimental::SOFIE::ROperator> op(p);\n";
            parserCode += "model.AddOperator(std::move(op));\n}\n";
         }
      }

      int batchSize = 1;
      if (inputShapes.size() > 0 && inputShapes[0].size() > 0) {
         batchSize = inputShapes[0][0];
         if (batchSize < 1) batchSize = 1;
      }
      if (verbose) std::cout << "generating the code with batch size = " << batchSize << " ...\n";

      parserCode += "model.Generate(TMVA::Experimental::SOFIE::Options::kDefault,"
                   + ROOT::Math::Util::ToString(batchSize) + ", 0, " + std::to_string(verbose) + "); \n";

      if (verbose) {
         parserCode += "model.PrintRequiredInputTensors();\n";
         parserCode += "model.PrintIntermediateTensors();\n";
         parserCode += "model.PrintOutputTensors();\n";
      }

      // add custom operators if needed
#if 0
      if (fCustomOperators.size() > 0) {
         if (verbose) {
            parserCode += "model.PrintRequiredInputTensors();\n";
            parserCode += "model.PrintIntermediateTensors();\n";
            parserCode += "model.PrintOutputTensors();\n";
         }
         for (auto & op : fCustomOperators) {
            parserCode += "{ auto p = new TMVA::Experimental::SOFIE::ROperator_Custom<float>(\""
                      + op.fOpName + "\"," + op.fInputNames + "," + op.fOutputNames + "," + op.fOutputShapes + ",\"" + op.fFileName + "\");\n";
            parserCode += "std::unique_ptr<TMVA::Experimental::SOFIE::ROperator> op(p);\n";
            parserCode += "model.AddOperator(std::move(op));\n}\n";
         }
         parserCode += "model.Generate(TMVA::Experimental::SOFIE::Options::kDefault,"
                   + ROOT::Math::Util::ToString(batchSize) + "); \n";
      }
#endif
      if (verbose > 1)
         parserCode += "model.PrintGenerated(); \n";
      parserCode += "model.OutputGenerated();\n";

      parserCode += "int nInputs = model.GetInputTensorNames().size();\n";

      // need information on number of inputs (assume output is 1)

      //end of parsing code, close the scope and return 1 to indicate a success
      parserCode += "return nInputs;\n}\n";

      if (verbose) std::cout << "//ParserCode being executed:\n" << parserCode << std::endl;

      auto iret = gROOT->ProcessLine(parserCode.c_str());
      if (iret <= 0) {
         std::string msg = "RSofieReader: error processing the parser code: \n" + parserCode;
         throw std::runtime_error(msg);
      }
      fNInputs = iret;
      if (fNInputs > 3) {
         throw std::runtime_error("RSofieReader does not yet support model with > 3 inputs");
      }

      // compile now the generated code and create Session class
      std::string modelHeader = modelName + ".hxx";
      if (verbose) std::cout << "compile generated code from file " <<modelHeader << std::endl;
      if (gSystem->AccessPathName(modelHeader.c_str())) {
         std::string msg = "RSofieReader: input header file " + modelHeader + " is not existing";
         throw std::runtime_error(msg);
      }
      if (verbose) std::cout << "Creating Inference function for model " << modelName << std::endl;
      std::string declCode;
      declCode += "#pragma cling optimize(2)\n";
      declCode += "#include \"" + modelHeader + "\"\n";
      // create global session instance: use UUID to have an unique name
      std::string sessionClassName = "TMVA_SOFIE_" + modelName + "::Session";
      TUUID uuid;
      std::string uidName = uuid.AsString();
      uidName.erase(std::remove_if(uidName.begin(), uidName.end(),
         []( char const& c ) -> bool { return !std::isalnum(c); } ), uidName.end());

      std::string sessionName = "session_" + uidName;
      declCode += sessionClassName + " " + sessionName + ";";

      if (verbose) std::cout << "//global session declaration\n" << declCode << std::endl;

      bool ret = gInterpreter->Declare(declCode.c_str());
      if (!ret) {
         std::string msg = "RSofieReader: error compiling inference code and creating session class\n" + declCode;
         throw std::runtime_error(msg);
      }

      fSessionPtr = (void *) gInterpreter->Calc(sessionName.c_str());

      // define a function to be called for inference
      std::stringstream ifuncCode;
      std::string funcName = "SofieInference_" + uidName;
      ifuncCode << "std::vector<float> " + funcName + "( void * ptr";
      for (int i = 0; i < fNInputs; i++)
         ifuncCode << ", float * data" << i;
      ifuncCode << ") {\n";
      ifuncCode << "   " << sessionClassName << " * s = " << "(" << sessionClassName << "*) (ptr);\n";
      ifuncCode << "   return s->infer(";
      for (int i = 0; i < fNInputs; i++) {
         if (i>0) ifuncCode << ",";
         ifuncCode << "data" << i;
      }
      ifuncCode << ");\n";
      ifuncCode << "}\n";

      if (verbose) std::cout << "//Inference function code using global session instance\n"
                              << ifuncCode.str() << std::endl;

      ret = gInterpreter->Declare(ifuncCode.str().c_str());
      if (!ret) {
         std::string msg = "RSofieReader: error compiling inference function\n" + ifuncCode.str();
         throw std::runtime_error(msg);
      }
      fFuncPtr = (void *) gInterpreter->Calc(funcName.c_str());
      //fFuncPtr = reinterpret_cast<std::vector<float> (*)(void *, const float *)>(fptr);
      fInitialized = true;
   }

   // Add custom operator
    void AddCustomOperator(const std::string &opName, const std::string &inputNames, const std::string & outputNames,
      const std::string & outputShapes, const std::string & fileName) {
         if (fInitialized)  std::cout << "WARNING: Model is already loaded and initialised. It must be done after adding the custom operators" << std::endl;
         fCustomOperators.push_back( {fileName, opName,inputNames, outputNames,outputShapes});
      }

   // implementations for different outputs
   std::vector<float> DoCompute(const std::vector<float> & x1) {
      if (fNInputs != 1) {
         std::string msg = "Wrong number of inputs - model requires " + std::to_string(fNInputs);
         throw std::runtime_error(msg);
      }
      auto fptr = reinterpret_cast<std::vector<float> (*)(void *, const float *)>(fFuncPtr);
      return fptr(fSessionPtr, x1.data());
   }
   std::vector<float> DoCompute(const std::vector<float> & x1, const std::vector<float> & x2) {
      if (fNInputs != 2) {
         std::string msg = "Wrong number of inputs - model requires " + std::to_string(fNInputs);
         throw std::runtime_error(msg);
      }
      auto fptr = reinterpret_cast<std::vector<float> (*)(void *, const float *, const float *)>(fFuncPtr);
      return fptr(fSessionPtr, x1.data(),x2.data());
   }
   std::vector<float> DoCompute(const std::vector<float> & x1, const std::vector<float> & x2, const std::vector<float> & x3) {
      if (fNInputs != 3) {
         std::string msg = "Wrong number of inputs - model requires " + std::to_string(fNInputs);
         throw std::runtime_error(msg);
      }
      auto fptr = reinterpret_cast<std::vector<float> (*)(void *, const float *, const float *, const float *)>(fFuncPtr);
      return fptr(fSessionPtr, x1.data(),x2.data(),x3.data());
   }

   /// Compute model prediction on vector
   template<typename... T>
   std::vector<float> Compute(T... x)
   {
      if(!fInitialized) {
         return std::vector<float>();
      }

      // Take lock to protect model evaluation
      R__WRITE_LOCKGUARD(ROOT::gCoreMutex);

      // Evaluate TMVA model (need to add support for multiple outputs)
      return DoCompute(x...);

   }
   std::vector<float> Compute(const std::vector<float> &x) {
      if(!fInitialized) {
         return std::vector<float>();
      }

      // Take lock to protect model evaluation
      R__WRITE_LOCKGUARD(ROOT::gCoreMutex);

      // Evaluate TMVA model (need to add support for multiple outputs)
      return DoCompute(x);
   }
   /// Compute model prediction on input RTensor
   /// The shape of the input tensor should be {nevents, nfeatures}
   /// and the return shape will be {nevents, noutputs}
   /// support for now only a single input
   RTensor<float> Compute(RTensor<float> &x)
   {
      if(!fInitialized) {
         return RTensor<float>({0});
      }
      const auto nrows = x.GetShape()[0];
      const auto rowsize = x.GetStrides()[0];
      auto fptr = reinterpret_cast<std::vector<float> (*)(void *, const float *)>(fFuncPtr);
      auto result = fptr(fSessionPtr, x.GetData());

      RTensor<float> y({nrows, result.size()}, MemoryLayout::ColumnMajor);
      std::copy(result.begin(),result.end(), y.GetData());
      //const bool layout = x.GetMemoryLayout() == MemoryLayout::ColumnMajor ? false : true;
      // assume column major layout
      for (size_t i = 1; i < nrows; i++) {
         result = fptr(fSessionPtr, x.GetData() + i*rowsize);
         std::copy(result.begin(),result.end(), y.GetData() + i*result.size());
      }
      return y;
   }

private:

   bool fInitialized = false;
   int fNInputs = 0;
   void * fSessionPtr = nullptr;
   void * fFuncPtr = nullptr;

   // data to insert custom operators
   struct CustomOperatorData {
      std::string fFileName; // code implementing the custom operator
      std::string fOpName; // operator name
      std::string fInputNames;  // input tensor names (convert as string as {"n1", "n2"})
      std::string fOutputNames;  // output tensor names converted as trind
      std::string fOutputShapes; // output shapes
   };
   std::vector<CustomOperatorData> fCustomOperators;

};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RREADER
