#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include<Mpi.h>

using namespace ROOT::Mpi;

using namespace TMVA;

void classification()
{
   Tools::Instance();

   TEnvironment env;

   
   if(COMM_WORLD.GetSize()==1) return; //needed to run ROOT tutorials in tests

   
   auto rank = COMM_WORLD.GetRank();
   if (COMM_WORLD.GetSize() != 4) {
      Error("classification", "Please run wih 4 processors.");
      COMM_WORLD.Abort(1);
   }

   auto outputFile = TFile::Open(Form("TMVA%d.root", rank), "RECREATE");

   Factory factory("TMVAClassification", outputFile,
                   "!V:ROC:Silent:Color:!DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification");

   DataLoader dataloader("dataset");

   if (COMM_WORLD.IsMainProcess()) {

      if (gSystem->AccessPathName("./tmva_class_example.root"))    // file does not exist in local directory
         gSystem->Exec("curl -O http://root.cern.ch/files/tmva_class_example.root");

      TFile *input = TFile::Open("./tmva_class_example.root");


      TTree *signalTree     = (TTree *)input->Get("TreeS");
      TTree *background     = (TTree *)input->Get("TreeB");

      dataloader.AddVariable("myvar1 := var1+var2", 'F');
      dataloader.AddVariable("myvar2 := var1-var2", "Expression 2", "", 'F');
      dataloader.AddVariable("var3",                "Variable 3", "units", 'F');
      dataloader.AddVariable("var4",                "Variable 4", "units", 'F');


      dataloader.AddSpectator("spec1 := var1*2",  "Spectator 1", "units", 'F');
      dataloader.AddSpectator("spec2 := var1*3",  "Spectator 2", "units", 'F');


      // global event weights per tree (see below for setting event-wise weights)
      Double_t signalWeight     = 1.0;
      Double_t backgroundWeight = 1.0;

      // You can add an arbitrary number of signal or background trees
      dataloader.AddSignalTree(signalTree,     signalWeight);
      dataloader.AddBackgroundTree(background, backgroundWeight);

      dataloader.SetBackgroundWeightExpression("weight");

      dataloader.PrepareTrainingAndTestTree("", "",
                                            "nTrain_Signal=4000:nTrain_Background=4000:SplitMode=Random:NormMode=NumEvents:!V");

   }

   COMM_WORLD.Bcast(dataloader, COMM_WORLD.GetMainProcess());

   if (rank == 0) {
      factory.BookMethod(&dataloader, TMVA::Types::kMLP, "MLP",
			 "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:!UseRegulator");
   }
   if (rank == 1) {
      factory.BookMethod(&dataloader, TMVA::Types::kBDT, "BDT",
                         "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5");
   }
   if (rank == 2) {
      factory.BookMethod(&dataloader, TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm");
   }
   if (rank == 3) {
      factory.BookMethod(&dataloader, TMVA::Types::kKNN, "KNN",
                         "H:nkNN=20:ScaleFrac=0.8:SigmaFact=1.0:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim");
   }

   factory.TrainAllMethods();

   factory.TestAllMethods();

   factory.EvaluateAllMethods();
   cout.flush();

   outputFile->Close();
}
