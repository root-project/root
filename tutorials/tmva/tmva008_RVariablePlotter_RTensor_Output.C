//
/// \file
/// \ingroup tutorial_tmva
/// Authors: Simone Azeglio, Lorenzo Moneta, Sitong An, Stefan Wunsch
/// This tutorial shows how to plot the distribution of variables from RTensor Outputs, after training a model
///

#include "TMVA/RVariablePlotter.h"
#include "TMVA/tmvaglob.h"

using namespace TMVA::Experimental;

// from tmva003_RReader.C
void train(const std::string &filename)
{
    // Create factory
    auto output = TFile::Open("TMVA.root", "RECREATE");
    auto factory = new TMVA::Factory("tmva003",
            output, "!V:!DrawProgressBar:AnalysisType=Classification");

    // Open trees with signal and background events
    auto data = TFile::Open(filename.c_str());
    auto signal = (TTree *)data->Get("TreeS");
    auto background = (TTree *)data->Get("TreeB");

    // Add variables and register the trees with the dataloader
    auto dataloader = new TMVA::DataLoader("tmva003_BDT");
    const std::vector<std::string> vars = {"var1", "var2", "var3", "var4"};
    for (const auto &var : vars) {
        dataloader->AddVariable(var);
    }
    dataloader->AddSignalTree(signal, 1.0);
    dataloader->AddBackgroundTree(background, 1.0);
    dataloader->PrepareTrainingAndTestTree("", "");

    // Train a TMVA method
    factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT", "!V:!H:NTrees=300:MaxDepth=2");
    factory->TrainAllMethods();
}



void tmva008_RVariablePlotter_RTensor_Output()
{   
    const std::string filename = "http://root.cern.ch/files/tmva_class_example.root";
    train(filename);

    // Next, we load the model from the TMVA XML file.
    RReader model("tmva003_BDT/weights/tmva003_BDT.weights.xml");

    // features (columns of TTree)
    auto ftrs = model.GetVariableNames(); 

    ROOT::RDataFrame sigDf("TreeS", filename);
    auto xSig = AsTensor<float>(sigDf, ftrs);
    auto ySig = model.Compute(xSig);

    ROOT::RDataFrame bkgDf("TreeB", filename);
    auto xBkg = AsTensor<float>(bkgDf, ftrs);
    auto yBkg = model.Compute(xBkg);

    
    // Create a variable plotter object giving the Tensors and the class labels.
    TMVA::RVariablePlotter plotter({xSig, xBkg}, {"Input Signal", "Input Background"}); //inputs


    TCanvas *c = new TCanvas("Input Variables", "Input Variables", 1400, 800);
    c->Divide(3, 2); // you can customize it wrt how many vars you have

    // legend vertices (default values)
    float minX = 0.7;
    float minY = 0.8;
    float maxX = 0.9;
    float maxY = 0.9;

    // use tmva style in DrawTensor() or custom style 
    bool useTMVAStyle = true; 

    for (unsigned int i = 0; i < ftrs.size(); i++) {
        c->cd(i + 1);
        c->Update();
        plotter.DrawTensor(ftrs[i], ftrs, useTMVAStyle);
        plotter.DrawLegend(minX, minY, maxX, maxY);
    }
    
    TCanvas *c1 = new TCanvas("Output", "Output", 1400, 800);
    //c->Divide(3, 2);

    TMVA::RVariablePlotter plotter_output({ySig, yBkg}, {"Output Signal", "Output Background"}); //outputs
    const std::vector<std::string> plot_vars = {"BDT_Score"}; // you can add models here we've only trained BDT

    for (unsigned int i = 0; i < plot_vars.size(); i++) {
        c1->cd(i + 1);
        c1->Update();
        plotter_output.DrawTensor(plot_vars[i], plot_vars, useTMVAStyle);
        plotter_output.DrawLegend(minX, minY, maxX, maxY);
    }
    

}

