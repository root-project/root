/// \file
/// \ingroup tutorial_ml
/// \notebook
/// Classification example of TMVA based on public Higgs UCI dataset
///
///  The UCI data set is a public HIGGS data set , see http://archive.ics.uci.edu/ml/datasets/HIGGS
/// used in this paper: Baldi, P., P. Sadowski, and D. Whiteson. “Searching for Exotic Particles in High-energy Physics
///                     with Deep Learning.” Nature Communications 5 (July 2, 2014).
///
/// \macro_image
/// \macro_code
/// \macro_output
///
/// \author Lorenzo Moneta

/***
## Declare Factory

Create the Factory class. Later you can choose the methods
whose performance you'd like to investigate.

The factory is the major TMVA object you have to interact with. Here is the list of parameters you need to pass

 - The first argument is the base of the name of all the output
weightfiles in the directory weight/ that will be created with the
method parameters

 - The second argument is the output file for the training results

 - The third argument is a string option defining some general configuration for the TMVA session. For example all TMVA output can be suppressed by removing the "!" (not) in front of the "Silent" argument in the option string

**/

void TMVA_Higgs_Classification() {

   // options to control used methods

   bool useLikelihood = true;    // likelihood based discriminant
   bool useLikelihoodKDE = false;    // likelihood based discriminant
   bool useFischer = true;       // Fischer discriminant
   bool useMLP = false;          // Multi Layer Perceptron
   bool useBDT = true;           // Boosted Decision Tree

   TMVA::Tools::Instance();

   auto outputFile = TFile::Open("Higgs_ClassificationOutput.root", "RECREATE");

   TMVA::Factory factory("TMVA_Higgs_Classification", outputFile,
                         "!V:ROC:!Silent:Color:AnalysisType=Classification" );

/**

## Setup Dataset(s)

Define now input data file and signal and background trees

 **/

   TString inputFileName = gROOT->GetTutorialDir() + "/machine_learning/data/Higgs_data.root";

   TFile *inputFile = nullptr;

   if (!gSystem->AccessPathName(inputFileName)) {
      // file exists
      inputFile = TFile::Open( inputFileName );
   }
   if (!inputFile) {
      Error("TMVA_Higgs_Classification","Input file is not found - exit");
      return;
   }

// --- Register the training and test trees

   TTree *signalTree     = (TTree*)inputFile->Get("sig_tree");
   TTree *backgroundTree = (TTree*)inputFile->Get("bkg_tree");

   signalTree->Print();

/***
## Declare DataLoader(s)

The next step is to declare the DataLoader class that deals with input variables

Define the input variables that shall be used for the MVA training
note that you may also use variable expressions, which can be parsed by TTree::Draw( "expression" )]

***/

   TMVA::DataLoader * loader = new TMVA::DataLoader("dataset");

   loader->AddVariable("m_jj");
   loader->AddVariable("m_jjj");
   loader->AddVariable("m_lv");
   loader->AddVariable("m_jlv");
   loader->AddVariable("m_bb");
   loader->AddVariable("m_wbb");
   loader->AddVariable("m_wwbb");

/// We set now the input data trees in the TMVA DataLoader class

// global event weights per tree (see below for setting event-wise weights)
   Double_t signalWeight     = 1.0;
   Double_t backgroundWeight = 1.0;

// You can add an arbitrary number of signal or background trees
   loader->AddSignalTree    ( signalTree,     signalWeight     );
   loader->AddBackgroundTree( backgroundTree, backgroundWeight );


// Set individual event weights (the variables must exist in the original TTree)
//    for signal    : factory->SetSignalWeightExpression    ("weight1*weight2");
//    for background: factory->SetBackgroundWeightExpression("weight1*weight2");
//loader->SetBackgroundWeightExpression( "weight" );

// Apply additional cuts on the signal and background samples (can be different)
   TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

// Tell the factory how to use the training and testing events
//
// If no numbers of events are given, half of the events in the tree are used
// for training, and the other half for testing:
//    loader->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
// To also specify the number of testing events, use:

   loader->PrepareTrainingAndTestTree( mycuts, mycutb,
                                       "nTrain_Signal=7000:nTrain_Background=7000:SplitMode=Random:NormMode=NumEvents:!V" );

/***
## Booking Methods

Here we book the TMVA methods. We book a Likelihood based on KDE (Kernel Density Estimation), a Fischer discriminant, a BDT
and a shallow neural network

 */


// Likelihood ("naive Bayes estimator")
if (useLikelihood) {
   factory.BookMethod(loader, TMVA::Types::kLikelihood, "Likelihood",
                           "H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );
}
// Use a kernel density estimator to approximate the PDFs
if (useLikelihoodKDE) {
   factory.BookMethod(loader, TMVA::Types::kLikelihood, "LikelihoodKDE",
                      "!H:!V:!TransformOutput:PDFInterpol=KDE:KDEtype=Gauss:KDEiter=Adaptive:KDEFineFactor=0.3:KDEborder=None:NAvEvtPerBin=50" );

}

// Fisher discriminant (same as LD)
if (useFischer) {
   factory.BookMethod(loader, TMVA::Types::kFisher, "Fisher", "H:!V:Fisher:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );
}

//Boosted Decision Trees
if (useBDT) {
   factory.BookMethod(loader,TMVA::Types::kBDT, "BDT",
                      "!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );
}

//Multi-Layer Perceptron (Neural Network)
if (useMLP) {
   factory.BookMethod(loader, TMVA::Types::kMLP, "MLP",
                      "!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
}


   /**
## Train Methods

Here we train all the previously booked methods.

    */

   factory.TrainAllMethods();

/**
   ## Test  all methods

 Now we test and evaluate all methods using the test data set
*/

   factory.TestAllMethods();

   factory.EvaluateAllMethods();

/// after we get the ROC curve and we display

   auto c1 = factory.GetROCCurve(loader);
   c1->Draw();

/// at the end we close the output file which contains the evaluation result of all methods and it can be used by TMVAGUI
/// to display additional plots

   outputFile->Close();


}
