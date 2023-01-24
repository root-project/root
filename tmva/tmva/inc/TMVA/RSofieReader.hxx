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
   /// Create TMVA model from ONNX file
   /// print level can be 0 (minimal) 1 with info , 2 with all ONNX parsing info
    RSofieReader(const std::string &path, std::vector<std::vector<size_t>> inputShape = {}, int verbose = 0)
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
         parserCode += "{\nTMVA::Experimental::SOFIE::RModel model = TMVA::Experimental::SOFIE::PyKeras::Parse(\"" + path + "\"); \n";
      }
      else if (type == kPt) {
         // use PyTorch direct parser
         if (gSystem->Load("libPyMVA") < 0) {
            throw std::runtime_error("RSofieReader: cannot use SOFIE with PyTorch since libPyMVA is missing");
         }
         if (inputShape.size() == 0) {
            throw std::runtime_error("RSofieReader: cannot use SOFIE with PyTorch since the input tensor shape is missing and is needed by the PyTorch parser");
         }
         std::string inputShapeStr = "{";
         for (unsigned int i = 0; i < inputShape.size(); i++) {
            inputShapeStr += "{ ";
            for (unsigned int j = 0; j < inputShape[i].size(); j++) {
               inputShapeStr += ROOT::Math::Util::ToString(inputShape[i][j]);
               if (j < inputShape[i].size()-1) inputShapeStr += ", ";
            }
            inputShapeStr += "}";
            if (i < inputShape.size()-1) inputShapeStr += ", ";
         }
         inputShapeStr += "}";
         parserCode += "{\nTMVA::Experimental::SOFIE::RModel model = TMVA::Experimental::SOFIE::PyTorch::Parse(\"" + path + "\", "
                    + inputShapeStr + "); \n";
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

      int batchSize = 1;
      if (inputShape.size() > 0 && inputShape[0].size() > 0) {
         batchSize = inputShape[0][0];
         if (batchSize < 1) batchSize = 1;
      }
      if (verbose) std::cout << "generating the code with batch size = " << batchSize << " ...\n";
      parserCode += "model.Generate(TMVA::Experimental::SOFIE::Options::kDefault,"
                   + ROOT::Math::Util::ToString(batchSize) + "); \n";
      if (verbose > 1)
         parserCode += "model.PrintGenerated(); \n";
      parserCode += "model.OutputGenerated();\n";

      //end of parsing code, close the scope and return 1 to indicate a success
      parserCode += "return 1;\n }\n";

      if (verbose) std::cout << "//ParserCode being executed:\n" << parserCode << std::endl;

      auto iret = gROOT->ProcessLine(parserCode.c_str());
      if (iret != 1) {
         std::string msg = "RSofieReader: error processing the parser code: \n" + parserCode;
         throw std::runtime_error(msg);
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

      fSessionPtr = (void*) gInterpreter->Calc(sessionName.c_str());

      // define a function to be called for inference
      std::stringstream ifuncCode;
      std::string funcName = "SofieInference_" + uidName;
      ifuncCode << "std::vector<float> " + funcName + "( void * ptr, float * data) {\n";
      ifuncCode << "   " << sessionClassName << " * s = " << "(" << sessionClassName << "*) (ptr);\n";
      ifuncCode << "   return s->infer(data);\n";
      ifuncCode << "}\n";

      if (verbose) std::cout << "//Inference function code using global session instance\n"
                              << ifuncCode.str() << std::endl;

      ret = gInterpreter->Declare(ifuncCode.str().c_str());
      if (!ret) {
         std::string msg = "RSofieReader: error compiling inference function\n" + ifuncCode.str();
         throw std::runtime_error(msg);
      }
      auto fptr = gInterpreter->Calc(funcName.c_str());
      fFuncPtr = reinterpret_cast<std::vector<float> (*)(void *, const float *)>(fptr);
      fInitialized = true;
   }

   /// Compute model prediction on vector
   std::vector<float> Compute(const std::vector<float> &x)
   {
      if(!fInitialized) {
         return std::vector<float>();
      }

      // Take lock to protect model evaluation
      R__WRITE_LOCKGUARD(ROOT::gCoreMutex);

      // Evaluate TMVA model (need to add support for multiple outputs)
      auto result =  fFuncPtr(fSessionPtr, x.data());
      return result;

   }
   /// Compute model prediction on input RTensor
   /// The shape of the input tensor should be {nevents, nfeatures}
   /// and the return shape will be {nevents, noutputs}
   RTensor<float> Compute(RTensor<float> &x)
   {
      if(!fInitialized) {
         return RTensor<float>({0});
      }
      const auto nrows = x.GetShape()[0];
      const auto rowsize = x.GetStrides()[0];
      auto result = fFuncPtr(fSessionPtr, x.GetData());

      RTensor<float> y({nrows, result.size()}, MemoryLayout::ColumnMajor);
      std::copy(result.begin(),result.end(), y.GetData());
      //const bool layout = x.GetMemoryLayout() == MemoryLayout::ColumnMajor ? false : true;
      // assume column major layout
      for (size_t i = 1; i < nrows; i++) {
         result = fFuncPtr(fSessionPtr, x.GetData() + i*rowsize);
         std::copy(result.begin(),result.end(), y.GetData() + i*result.size());
      }
      return y;
   }

private:

   bool fInitialized = false;
   void * fSessionPtr = nullptr;
   std::function<std::vector<float> (void *, const float *)> fFuncPtr;

};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RREADER
