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
import torch\n\
from torch import nn\n\n\
# Define model\n\
model = nn.Sequential(\n\
                nn.Linear(4, 64),\n\
                nn.ReLU(),\n\
                nn.Linear(64, 2),\n\
                nn.Softmax(dim=1)\n)\n\n\
# Construct loss function and Optimizer.\n\
criterion = torch.nn.MSELoss()\n\
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)\n\n\
def fit(model, train_X, train_y, num_epochs, batch_size, optimizer, criterion):\n\
    # Set to train mode\n\
    model.train()\n\n\
    train_X = torch.from_numpy(train_X)\n\
    train_y = torch.from_numpy(train_y)\n\n\
    # Prepare Dummy DataLoader (NOTE: Inefficient, will be replaced later)\n\
    # TODO: Use original PyTorch Dataloader\n\
    X, y = [], []\n\
    for i in range(0, train_X.shape[0], batch_size):\n\
        X.append(train_X[i:i + batch_size])\n\
        y.append(train_y[i:i + batch_size])\n\
    data = list(zip(X, y))\n\n\
    for epoch in range(num_epochs):\n\
        running_loss = 0.0\n\
        for i, (X, y) in enumerate(data):\n\
            optimizer.zero_grad()\n\
            output = model(X)\n\
            loss  = criterion(output, y)\n\
            loss.backward()\n\
            optimizer.step()\n\n\
            # print statistics\n\
            running_loss += loss.item()\n\
            if i % 8 == 7:    # print every 8 mini-batches\n\
                print(f\"[{epoch+1}, {i+1}] loss: {running_loss / 8 :.3f}\")\n\
                running_loss = 0.0\n\n\
    print(f\"Finished Training on Epoch: {epoch+1}!\")\n\n\n\
load_model_custom_objects = {\"optimizer\": optimizer, \"criterion\": criterion, \"train_func\": fit}\n\n\
# Store model to file\n\
m = torch.jit.script(model)\n\
torch.jit.save(m, \"PyTorchModelClassification.pt\")";


int testPyTorchClassification(){
   // Get data file
   std::cout << "Get test data..." << std::endl;
   TString fname = "./tmva_class_example.root";
   if (gSystem->AccessPathName(fname))  // file does not exist in local directory
      gSystem->Exec("curl -O http://root.cern.ch/files/tmva_class_example.root");
   TFile *input = TFile::Open(fname);

   // Build model from python file
   std::cout << "Generate PyTorch model..." << std::endl;
   UInt_t ret;
   ret = gSystem->Exec("echo '"+pythonSrc+"' > generatePyTorchModelClassification.py");
   if(ret!=0){
       std::cout << "[ERROR] Failed to write python code to file" << std::endl;
       return 1;
   }
   ret = gSystem->Exec("python generatePyTorchModelClassification.py");
   if(ret!=0){
       std::cout << "[ERROR] Failed to generate model using python" << std::endl;
       return 1;
   }

   // // Setup PyMVA and factory
   std::cout << "Setup TMVA..." << std::endl;
   TMVA::PyMethodBase::PyInitialize();
   TFile* outputFile = TFile::Open("ResultsTestPyTorchClassification.root", "RECREATE");
   TMVA::Factory *factory = new TMVA::Factory("testPyTorchClassification", outputFile,
      "!V:Silent:Color:!DrawProgressBar:AnalysisType=Classification");

   // Load data
   TMVA::DataLoader *dataloader = new TMVA::DataLoader("datasetTestPyTorchClassification");

   TTree *signal = (TTree*)input->Get("TreeS");
   TTree *background = (TTree*)input->Get("TreeB");
   dataloader->AddSignalTree(signal);
   dataloader->AddBackgroundTree(background);

   dataloader->AddVariable("var1");
   dataloader->AddVariable("var2");
   dataloader->AddVariable("var3");
   dataloader->AddVariable("var4");

   dataloader->PrepareTrainingAndTestTree("",
      "SplitMode=Random:NormMode=NumEvents:!V");

   // Book and train method
   factory->BookMethod(dataloader, TMVA::Types::kPyTorch, "PyTorch",
      "!H:!V:VarTransform=D,G:FilenameModel=PyTorchModelClassification.pt:FilenameTrainedModel=trainedPyTorchModelClassification.pt:NumEpochs=10:BatchSize=32:UserCode=generatePyTorchModelClassification.py");
   std::cout << "Training model..." << std::endl;
   factory->TrainAllMethods();

   // Clean-up
   delete factory;
   delete dataloader;
   delete outputFile;

   // TODO: Verify response and setup reader

   return 0;
}

int main(){
   int err = testPyTorchClassification();
   return err;
}
