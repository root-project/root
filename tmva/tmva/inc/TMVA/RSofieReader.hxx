/**********************************************************************************
 * Project: ROOT - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Authors:                                                                       *
 *      Lorenzo Moneta                                  *
 *                                                                                *
 * Copyright (c) 2022:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
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
    RSofieReader(const std::string &path, int verbose = 0)
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
         Error("RSofieReader","Input file is not an ONNX or Keras or PyTorch file");
         return;
      }
      if (pos1 == std::string::npos)
         pos1 = 0;
      else
         pos1 += 1;
      std::string modelName = path.substr(pos1,pos2-pos1);
      std::string fileType = path.substr(pos2+1, path.length()-pos2-1);
      if (verbose) std::cout << "Parsing SOFIE model " << modelName << " of type " << fileType << std::endl;


      std::string parserCode;
      if (type == kONNX) {
         // check first if we can load the SOFIE parser library
         if (gSystem->Load("libROOTTMVASofieParser") < 0) {
            Error("RSofieReader","Cannot use SOFIE with ONNX since libROOTTMVASofieParser is missing");
            return;
         }
         gInterpreter->Declare("#include \"TMVA/RModelParser_ONNX.hxx\"");
         parserCode += "TMVA::Experimental::SOFIE::RModelParser_ONNX parser; \n";
         if (verbose == 2)
            parserCode += "TMVA::Experimental::SOFIE::RModel model = parser.Parse(\"" + path + "\",true); \n";
         else
            parserCode += "TMVA::Experimental::SOFIE::RModel model = parser.Parse(\"" + path + "\"); \n";
      }
      else if (type == kKeras) {
         // use Keras direct parser
         if (gSystem->Load("libPyMVA") < 0) {
            Error("RSofieReader","Cannot use SOFIE with Keras since libPyMVA is missing");
            return;
         }
         parserCode += "TMVA::Experimental::SOFIE::RModel model = SOFIE::PyKeras::Parse(\"" + path + "\"); \n";
      }
      else if (type == kPt) {
         // use PyTorch direct parser
         if (gSystem->Load("libPyMVA") < 0) {
            Error("RSofieReader","Cannot use SOFIE with PyTorch since libPyMVA is missing");
            return;
         }
         parserCode += "TMVA::Experimental::SOFIE::RModel model = SOFIE::PyTorch::Parse(\"" + path + "\"); \n";
      }
      else if (type == kROOT) {
         // use  parser from ROOT
         parserCode += "auto fileRead = TFile::Open(\"" + path + "\",\"READ\");\n";
         parserCode += "TMVA::Experimental::SOFIE::RModel * modelPtr;\n";
         parserCode += "auto keyList = fileRead->GetListOfKeys(); TString name;\n";
         parserCode += "for (const auto&& k : *keyList)  { \n";
         parserCode += "   TString cname =  ((TKey*)k)->GetClassName();  if (cname==\"TMVA::Experimental::SOFIE::RModel\") name = k->GetName(); }\n";
         parserCode += "fileRead->GetObject(name,modelPtr); fileRead->Close(); delete fileRead;\n";
         parserCode += "TMVA::Experimental::SOFIE::RModel & model = *modelPtr;\n";
      }

      if (verbose) std::cout << parserCode << std::endl;
      gROOT->ProcessLine(parserCode.c_str());

      if (verbose) std::cout << "generating the code ...\n";
      parserCode = "model.Generate(); \n";
      if (verbose)
         parserCode += "model.PrintGenerated(); \n";
      parserCode += "model.OutputGenerated();\n";

      if (verbose) std::cout << parserCode << std::endl;

      gROOT->ProcessLine(parserCode.c_str());

      if (verbose) std::cout << "compile generated code \n";

      // compile now the generated code and create Session class
      std::string modelHeader = modelName + ".hxx";
      if (gSystem->AccessPathName(modelHeader.c_str())) {
         //std::string msg = "input header file is not existing";
         Error("RSofieReader","Input header file %s is not existing",modelHeader.c_str());
         return;
      }
      if (verbose) std::cout << "Parsing SOFIE model " << modelName << std::endl;
      std::string declCode;
      declCode += "#pragma cling optimize(2)\n";
      declCode += "#include \"" + modelHeader + "\"\n";
      std::string sessionClassName = "TMVA_SOFIE_" + modelName + "::Session";
      TUUID uuid;
      std::string uidName = uuid.AsString();
      uidName.erase(std::remove_if(uidName.begin(), uidName.end(),
         []( char const& c ) -> bool { return !std::isalnum(c); } ), uidName.end());

      std::string sessionName = "session_" + uidName;
      declCode += sessionClassName + " " + sessionName + ";";

      if (verbose) std::cout << declCode << std::endl;

      gInterpreter->Declare(declCode.c_str());
      fInitialized = true;
      TClass * sessionClass = TClass::GetClass(sessionClassName.c_str());

      fSessionPtr = (void*) gInterpreter->Calc(sessionName.c_str());

      // define a function to be called for inference
      std::stringstream ifuncCode;
      std::string funcName = "SofieInference_" + uidName;
      ifuncCode << "std::vector<float> " + funcName + "( void * ptr, float * data) {\n";
      ifuncCode << sessionClassName << " * s = " << "(" << sessionClassName << "*) (ptr);\n";
      ifuncCode << "return s->infer(data);\n";
      ifuncCode << "}\n";

      if (verbose) std::cout << ifuncCode.str() << std::endl;

      gInterpreter->Declare(ifuncCode.str().c_str());
      auto fptr = gInterpreter->Calc(funcName.c_str());
      fFuncPtr = reinterpret_cast<std::vector<float> (*)(void *, const float *)>(fptr);
      //fFuncPtr = fptr2;
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
