// Author: Sanjiban Sengupta, 2021
// Description:
//           This is to test the Serialisation of RModel class
//           defined in SOFIE. The program is run when the
//           target 'TestCustomModelsFromROOT' is built. The program
//           generates the required .hxx file after reading a written
//           ROOT file which stores the object of the RModel class.

#include <iostream>

#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"
#include "TFile.h"

using namespace TMVA::Experimental::SOFIE;

int main(int argc, char *argv[]){
   if (argc < 2) {
      std::cerr << "[ERROR]: Missing ONNX input file\n";
      return -1;
   }
   std::string outname=argv[2];
   RModelParser_ONNX parser;
   RModel model = parser.Parse(argv[1]);
   TFile fileWrite((outname+"_FromROOT.root").c_str(),"RECREATE");
   model.Write("model");
   fileWrite.Close();
   TFile fileRead((outname+"_FromROOT.root").c_str(),"READ");
   RModel *modelPtr;
   fileRead.GetObject("model",modelPtr);
   fileRead.Close();
   // in this case we don't write session class and weight file
   modelPtr->Generate(false, false);
   modelPtr->OutputGenerated(outname+"_FromROOT.hxx");
   return 0;
}
