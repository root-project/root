/*
An example script to modify the interpolation used in HistFactory models.
Usage:

make the standard example workspace with histfactory
$ prepareHistFactory
$ hist2workspace config/example.xml

inspect file (note we are using the model with Gaussian constraints)
$ root.exe results/example_combined_GaussExample_model.root
root [1] combined->Print()

notice there is a new class in the new version:
RooStats::HistFactory::FlexibleInterpVar::background1_channel1_epsilon[ paramList=(alpha_syst2) ] = 0.999209

You only need this macro if you don't want the CMS style normalization interpolation.

root [2] .L $ROOTSYS/tutorials/histfactory/ModifyInterpolation.C+
root [3] CheckInterpolation(combined)
[#1] INFO:InputArguments -- interp code for alpha_syst1 = 1
[#1] INFO:InputArguments -- interp code for alpha_syst2 = 1
[#1] INFO:InputArguments -- interp code for alpha_syst3 = 1

One can change the interpolation for a specific set of parameters
root [5] combined->set("ModelConfig_NuisParams")->Print()
RooArgSet:: = (alpha_syst2,alpha_syst3)
root [6] ModifyInterpolationForSet(combined->set("ModelConfig_NuisParams"),2)
root [7] CheckInterpolation(combined)
[#1] INFO:InputArguments -- interp code for alpha_syst1 = 1
[#1] INFO:InputArguments -- interp code for alpha_syst2 = 2
[#1] INFO:InputArguments -- interp code for alpha_syst3 = 2

Or one can change the interpolation type for all the parameters:
root [11] ModifyInterpolationForAll(combined,2)
root [12] CheckInterpolation(combined)
[#1] INFO:InputArguments -- interp code for alpha_syst1 = 2
[#1] INFO:InputArguments -- interp code for alpha_syst2 = 2
[#1] INFO:InputArguments -- interp code for alpha_syst3 = 2

and then you can save the workspace with those modifications
root [13] combined->writeToFile("testModified.root")

then test to make sure the changes are in the file
$ root.exe testModified.root
root [1] .L $ROOTSYS/tutorials/histfactory/ModifyInterpolation.C+
root [2] CheckInterpolation(combined)
[#1] INFO:InputArguments -- interp code for alpha_syst1 = 2
[#1] INFO:InputArguments -- interp code for alpha_syst2 = 2
[#1] INFO:InputArguments -- interp code for alpha_syst3 = 2
*/
#include "RooRealVar.h"
#include "TIterator.h"
#include "RooWorkspace.h"
#include "RooStats/HistFactory/FlexibleInterpVar.h"

using namespace RooStats;
using namespace HistFactory;

// define functions for ACLIC
void ModifyInterpolationForAll(RooWorkspace* ws, int code=1);
void ModifyInterpolationForSet(RooArgSet* modifySet, int code = 1);
void CheckInterpolation(RooWorkspace* ws);

// Codes for interpolation
// code = 0: piece-wise linear
// code = 1: pice-wise log
// code = 2: parabolic interp with linear extrap
// code = 3: parabolic version of log-normal

void ModifyInterpolation(){
  cout <<"Choose from the following"<<endl;
  cout <<"void ModifyInterpolationForAll(RooWorkspace* ws, int code=1);"<<endl;
  cout <<"void ModifyInterpolationForSet(RooArgSet* modifySet, int code = 1);"<<endl;
  cout <<"void CheckInterpolation(RooWorkspace* ws);"<<endl;
}

void ModifyInterpolationForAll(RooWorkspace* ws, int code){
  RooArgSet funcs = ws->allFunctions();
  TIterator* it = funcs.createIterator();
  TObject* tempObj=0;
  while((tempObj=it->Next())){
    FlexibleInterpVar* flex = dynamic_cast<FlexibleInterpVar*>(tempObj);
    if(flex){
      flex->setAllInterpCodes(code);
    }
  }
}

void ModifyInterpolationForSet(RooArgSet* modifySet, int code){

  TIterator* it = modifySet->createIterator();
  RooRealVar* alpha=0;
  while((alpha=(RooRealVar*)it->Next())){
    TIterator* serverIt = alpha->clientIterator();
    TObject* tempObj=0;
    while((tempObj=serverIt->Next())){
      FlexibleInterpVar* flex = dynamic_cast<FlexibleInterpVar*>(tempObj);
      if(flex){
         flex->printAllInterpCodes();
         flex->setInterpCode(*alpha,code);
         flex->printAllInterpCodes();
      }
    }
  }

}


void CheckInterpolation(RooWorkspace* ws){
  RooArgSet funcs = ws->allFunctions();
  TIterator* it = funcs.createIterator();
  TObject* tempObj=0;
  while((tempObj=it->Next())){
    FlexibleInterpVar* flex = dynamic_cast<FlexibleInterpVar*>(tempObj);
    if(flex){
      flex->printAllInterpCodes();
    }
  }
}
