//
/// \file
/// \ingroup tutorial_tmva
/// Authors: Simone Azeglio, Lorenzo Moneta , Stefan Wunsch
/// This tutorial shows how to plot the distribution of variables.
///


#include "TMVA/RVariablePlotter.h"

using namespace TMVA::Experimental;


void tmva005_RVariablePlotter()
{
    // Initialize ROOT dataframes from signal and background datasets
    const std::string filename = "http://root.cern.ch/files/tmva_class_example.root";
    ROOT::RDataFrame sig1("TreeS", filename);
    ROOT::RDataFrame bkg1("TreeB", filename);

    // Apply transformations on the datasets to be included in the study
    //auto transform = [](auto df) { return df.Define("var5", "var1 * var2"); };
    auto transform = [](ROOT::RDF::RNode df) { return df.Define("var5", "var1 * var2"); };
    auto sig2 = transform(sig1);
    auto bkg2 = transform(bkg1);


    // Create a variable plotter object giving the dataframes and the class labels.
    TMVA::RVariablePlotter plotter({sig2, bkg2}, {"Signal", "Background"});

    // Create a canvas with four pads for plotting
    auto c = new TCanvas("", "", 1200, 800);
    c->Divide(3, 2);

    
    // legend vertices
    float minX = 0.7;
    float minY = 0.8;
    float maxX = 0.9;
    float maxY = 0.9;
    

    // Place plots on the pads of the canvas
    const std::vector<std::string> variables = {"var1", "var2", "var3", "var4", "var5"};
    
    for (unsigned int i = 0; i < variables.size(); i++) {
        c->cd(i + 1);
        c->Update();
        gPad->SetGrid(1,1); // plotting a background grid
        plotter.Draw(variables[i]);
        plotter.DrawLegend(minX, minY, maxX, maxY);
        
    }

}


