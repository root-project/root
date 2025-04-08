//
/// \file
/// \ingroup tutorial_tmva
/// Authors: Simone Azeglio, Lorenzo Moneta, Sitong An, Stefan Wunsch
/// This tutorial shows how to plot the distribution of variables from an output file of a classification problem, 
/// in this case we take into consideration "Higgs_ClassificatioOutput.root", coming from "TMVA_Higgs_Classification.C"

#include "TMVA/RVariablePlotter.h"
#include "TMVA/tmvaglob.h"

using namespace TMVA::Experimental;

void tmva010_RVariablePlotter_Higgs_MultiROC()
{
    // Initialize ROOT dataframes from signal and background datasets
    const std::string filename = "Higgs_ClassificationOutput.root";
    ROOT::RDataFrame testDf("dataset/TestTree", filename);

    // extracting signal & background
    auto sigDf  = testDf.Filter("classID==0");   
    auto bkgDf  = testDf.Filter("classID==1");   
    
    // columns (in this case we are looking for model outputs so from BDT till the end)
    // const std::vector<std::string> vars = sigDf.GetColumnNames(); 
    // { "classID", "className", "m_jj", "m_jjj", "m_lv", "m_jlv", 
    // "m_bb", "m_wbb", "m_wwbb", "weight", "BDT", "DNN_CPU", "Fisher", "Likelihood", "prob_Fisher" }

    // for example 
    const std::vector<std::string> vars = {"BDT", "DNN_CPU", "Fisher", "Likelihood"}; 
    
    // Create a variable plotter object giving the Tensors and the class labels.
    TMVA::RVariablePlotter plotter({sigDf, bkgDf}, {"Signal", "Background"});
    
    TCanvas *c = new TCanvas("c", "c", 900, 600);
    
    // use tmva style in DrawMultiROCCurve() or custom style 
    bool useTMVAStyle = false; 

    plotter.DrawMultiROCCurve(vars, useTMVAStyle);
        
}