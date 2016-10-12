#include <iostream>

#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TMVA/Factory.h"
#include "TMVA/Reader.h"
#include "TMVA/DataLoader.h"
#include "TMVA/PyMethodBase.h"

TString pythonSrc = "\
from keras.models import Sequential\n\
from keras.layers.core import Dense, Activation\n\
from keras import initializations\n\
from keras.optimizers import SGD\n\
\n\
model = Sequential()\n\
model.add(Dense(64, init=\"normal\", activation=\"tanh\", input_dim=2))\n\
model.add(Dense(1, init=\"normal\", activation=\"linear\"))\n\
model.compile(loss=\"mean_squared_error\", optimizer=SGD(lr=0.01))\n\
model.save(\"kerasModelRegression.h5\")\n";

int testPyKerasRegression(){
   // Get data file
   TString fname = "./tmva_reg_example.root";
   if (gSystem->AccessPathName(fname))  // file does not exist in local directory
      gSystem->Exec("curl -O http://root.cern.ch/files/tmva_reg_example.root");
   TFile *input = TFile::Open(fname);

   // Build model from python file
   UInt_t ret;
   ret = gSystem->Exec("echo '"+pythonSrc+"' > generateKerasModelRegression.py");
   if(ret!=0){
       std::cout << "[ERROR] Failed to write python code to file" << std::endl;
       return 1;
   }
   ret = gSystem->Exec("python generateKerasModelRegression.py");
   if(ret!=0){
       std::cout << "[ERROR] Failed to generate model using python" << std::endl;
       return 1;
   }

   // Setup PyMVA and factory
   TMVA::PyMethodBase::PyInitialize();
   TFile* outputFile = TFile::Open("ResultsTestPyKerasRegression.root", "RECREATE");
   TMVA::Factory *factory = new TMVA::Factory("testPyKerasRegression", outputFile,
      "!V:Silent:Color:!DrawProgressBar:AnalysisType=Regression");

   // Load data
   TMVA::DataLoader *dataloader = new TMVA::DataLoader("datasetTestPyKerasRegression");

   TTree *tree = (TTree*)input->Get("TreeR");
   dataloader->AddRegressionTree(tree);

   dataloader->AddVariable("var1");
   dataloader->AddVariable("var2");
   dataloader->AddTarget("fvalue");

   dataloader->PrepareTrainingAndTestTree("",
      "SplitMode=Random:NormMode=NumEvents:!V");

   // Book and train method
   factory->BookMethod(dataloader, TMVA::Types::kPyKeras, "PyKeras",
      "!H:!V:VarTransform=D,G:FilenameModel=kerasModelRegression.h5:FilenameTrainedModel=trainedKerasModelRegression.h5:NumEpochs=10:BatchSize=32:SaveBestOnly=false:Verbose=0");
   factory->TrainAllMethods();

   // Clean-up
   delete factory;
   delete dataloader;
   delete outputFile;

   // Setup reader
   TMVA::Reader *reader = new TMVA::Reader("!Color:Silent");
   Float_t vars[3];
   reader->AddVariable("var1", vars+0);
   reader->AddVariable("var2", vars+1);
   reader->BookMVA("PyKeras", "datasetTestPyKerasRegression/weights/testPyKerasRegression_PyKeras.weights.xml");

   // Get mean squared error on events
   tree->SetBranchAddress("var1", vars+0);
   tree->SetBranchAddress("var2", vars+1);
   tree->SetBranchAddress("fvalue", vars+2);

   UInt_t numEvents = 100;
   Float_t meanMvaError = 0;
   for(UInt_t i=0; i<numEvents; i++){
      tree->GetEntry(i);
      meanMvaError += std::pow(vars[2]-reader->EvaluateMVA("PyKeras"),2);
   }
   meanMvaError = meanMvaError/float(numEvents);

   // Check whether the response is obviously better than guessing
   if(meanMvaError > 30.0){
      std::cout << "[ERROR] Mean squared error is " << meanMvaError << " (>30.0)" << std::endl;
      return 1;
   }

   return 0;
}

int main(){
   int err = testPyKerasRegression();
   return err;
}
