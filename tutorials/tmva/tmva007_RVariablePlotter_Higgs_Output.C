//
/// \file
/// \ingroup tutorial_tmva
/// Authors: Simone Azeglio, Lorenzo Moneta, Sitong An, Stefan Wunsch
/// This tutorial shows how to plot the distribution of variables from an output file of a classification problem, 
/// in this case we take into consideration "Higgs_ClassificatioOutput.root", coming from "TMVA_Higgs_Classification.C"

#include "TMVA/RVariablePlotter.h"
#include "TMVA/tmvaglob.h"

using namespace TMVA::Experimental;

void tmva007_RVariablePlotter_Higgs_Output()
{
    // Initialize ROOT dataframes from signal and background datasets
    const std::string filename = "Higgs_ClassificationOutput.root";
    ROOT::RDataFrame testDf("dataset/TestTree", filename);

    // extracting signal & background
    auto sigDf  = testDf.Filter("classID==0");   
    auto bkgDf  = testDf.Filter("classID==1");   
    
    // columns (you can plot the ones you prefer)
    // const std::vector<std::string> vars = sigDf.GetColumnNames(); 
    // { "classID", "className", "m_jj", "m_jjj", "m_lv", "m_jlv", 
    // "m_bb", "m_wbb", "m_wwbb", "weight", "BDT", "DNN_CPU", "Fisher", "Likelihood", "prob_Fisher" }

    // for example 
    const std::vector<std::string> vars = {"BDT", "DNN_CPU", "Fisher", "Likelihood"}; 
    
    // Create a variable plotter object giving the Tensors and the class labels.
    TMVA::RVariablePlotter plotter({sigDf, bkgDf}, {"Signal", "Background"});
    
    TCanvas *c = new TCanvas("c", "c", 1400, 800);
    c->Divide(3, 2);

    // legend vertices
    float minX = 0.7;
    float minY = 0.8;
    float maxX = 0.9;
    float maxY = 0.9;
    
    // use tmva style in Draw() or custom style 
    bool useTMVAStyle = true; 

    for (unsigned int i = 0; i < vars.size(); i++) {
        c->cd(i + 1);
        c->Update();
        plotter.Draw(vars[i], useTMVAStyle);
        plotter.DrawLegend(minX, minY, maxX, maxY);
    }

}



