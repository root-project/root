#include <iostream>

#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TMVA/Factory.h"
#include "TMVA/Reader.h"
#include "TMVA/DataLoader.h"
#include "TMVA/PyMethodBase.h"

TString pythonSrc = "import torch\n\
from torch import nn\n\
\n\
# Define model\n\
model = nn.Sequential(\n\
                nn.Linear(2, 64),\n\
                nn.Tanh(),\n\
                nn.Linear(64, 1))\n\
\n\
# Construct loss function and Optimizer.\n\
criterion = torch.nn.MSELoss()\n\
optimizer = torch.optim.SGD\n\
\n\
\n\
def fit(model, train_loader, val_loader, num_epochs, batch_size, optimizer, criterion, save_best, scheduler):\n\
    trainer = optimizer(model.parameters(), lr=0.01)\n\
    schedule, schedulerSteps = scheduler\n\
    best_val = None\n\
\n\
    for epoch in range(num_epochs):\n\
        # Training Loop\n\
        # Set to train mode\n\
        model.train()\n\
        running_train_loss = 0.0\n\
        running_val_loss = 0.0\n\
        for i, (X, y) in enumerate(train_loader):\n\
            trainer.zero_grad()\n\
            output = model(X)\n\
            train_loss = criterion(output, y)\n\
            train_loss.backward()\n\
            trainer.step()\n\
\n\
            # print train statistics\n\
            running_train_loss += train_loss.item()\n\
            if i % 32 == 31:    # print every 32 mini-batches\n\
                print(f\"[{epoch+1}, {i+1}] train loss: {running_train_loss / 32 :.3f}\")\n\
                running_train_loss = 0.0\n\
\n\
        if schedule:\n\
            schedule(optimizer, epoch, schedulerSteps)\n\
\n\
        # Validation Loop\n\
        # Set to eval mode\n\
        model.eval()\n\
        with torch.no_grad():\n\
            for i, (X, y) in enumerate(val_loader):\n\
                output = model(X)\n\
                val_loss = criterion(output, y)\n\
                running_val_loss += val_loss.item()\n\
\n\
            curr_val = running_val_loss / len(val_loader)\n\
            if save_best:\n\
               if best_val==None:\n\
                   best_val = curr_val\n\
               best_val = save_best(model, curr_val, best_val)\n\
\n\
            # print val statistics per epoch\n\
            print(f\"[{epoch+1}] val loss: {curr_val :.3f}\")\n\
            running_val_loss = 0.0\n\
\n\
    print(f\"Finished Training on {epoch+1} Epochs!\")\n\
\n\
    return model\n\
\n\
\n\
def predict(model, test_X, batch_size=32):\n\
    # Set to eval mode\n\
    model.eval()\n\
   \n\
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_X))\n\
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n\
\n\
    predictions = []\n\
    with torch.no_grad():\n\
        for i, data in enumerate(test_loader):\n\
            X = data[0]\n\
            outputs = model(X)\n\
            predictions.append(outputs)\n\
        preds = torch.cat(predictions)\n\
   \n\
    return preds.numpy()\n\
\n\
\n\
load_model_custom_objects = {\"optimizer\": optimizer, \"criterion\": criterion, \"train_func\": fit, \"predict_func\": predict}\n\
\n\
# Store model to file\n\
m = torch.jit.script(model)\n\
torch.jit.save(m,\"PyTorchModelRegression.pt\")\n";

int testPyTorchRegression(){
   // Get data file
   std::cout << "Get test data..." << std::endl;
   TString fname = "./tmva_reg_example.root";
   if (gSystem->AccessPathName(fname))  // file does not exist in local directory
      gSystem->Exec("curl -O http://root.cern.ch/files/tmva_reg_example.root");
   TFile *input = TFile::Open(fname);

   // Build model from python file
   std::cout << "Generate PyTorch model..." << std::endl;
   UInt_t ret;
   ret = gSystem->Exec("echo '"+pythonSrc+"' > generatePyTorchModelRegression.py");
   if(ret!=0){
       std::cout << "[ERROR] Failed to write python code to file" << std::endl;
       return 1;
   }
   ret = gSystem->Exec("python generatePyTorchModelRegression.py");
   if(ret!=0){
       std::cout << "[ERROR] Failed to generate model using python" << std::endl;
       return 1;
   }

   // Setup PyMVA and factory
   std::cout << "Setup TMVA..." << std::endl;
   TMVA::PyMethodBase::PyInitialize();
   TFile* outputFile = TFile::Open("ResultsTestPyTorchRegression.root", "RECREATE");
   TMVA::Factory *factory = new TMVA::Factory("testPyTorchRegression", outputFile,
      "!V:Silent:Color:!DrawProgressBar:AnalysisType=Regression");

   // Load data
   TMVA::DataLoader *dataloader = new TMVA::DataLoader("datasetTestPyTorchRegression");

   TTree *tree = (TTree*)input->Get("TreeR");
   dataloader->AddRegressionTree(tree);

   dataloader->AddVariable("var1");
   dataloader->AddVariable("var2");
   dataloader->AddTarget("fvalue");

   dataloader->PrepareTrainingAndTestTree("",
      "SplitMode=Random:NormMode=NumEvents:!V");

   // Book and train method
   factory->BookMethod(dataloader, TMVA::Types::kPyTorch, "PyTorch",
      "!H:!V:VarTransform=D,G:FilenameModel=PyTorchModelRegression.pt:FilenameTrainedModel=trainedPyTorchModelRegression.h5:NumEpochs=10:BatchSize=32:SaveBestOnly=false:UserCode=generatePyTorchModelRegression.py");
   std::cout << "Train model..." << std::endl;
   factory->TrainAllMethods();

   // Clean-up
   delete factory;
   delete dataloader;
   delete outputFile;

   // Setup reader
   UInt_t numEvents = 100;
   std::cout << "Run reader and estimate target of " << numEvents << " events..." << std::endl;
   TMVA::Reader *reader = new TMVA::Reader("!Color:Silent");
   Float_t vars[3];
   reader->AddVariable("var1", vars+0);
   reader->AddVariable("var2", vars+1);
   reader->BookMVA("PyTorch", "datasetTestPyTorchRegression/weights/testPyTorchRegression_PyTorch.weights.xml");

   // Get mean squared error on events
   tree->SetBranchAddress("var1", vars+0);
   tree->SetBranchAddress("var2", vars+1);
   tree->SetBranchAddress("fvalue", vars+2);

   Float_t meanMvaError = 0;
   for(UInt_t i=0; i<numEvents; i++){
      tree->GetEntry(i);
      meanMvaError += std::pow(vars[2]-reader->EvaluateMVA("PyTorch"),2);
   }
   meanMvaError = meanMvaError/float(numEvents);

   // Check whether the response is obviously better than guessing
   std::cout << "Mean squared error: " << meanMvaError << std::endl;
   if(meanMvaError > 30.0){
      std::cout << "[ERROR] Mean squared error is " << meanMvaError << " (>30.0)" << std::endl;
      return 1;
   }

   return 0;
}

int main(){
   int err = testPyTorchRegression();
   return err;
}
