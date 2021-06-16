// @(#)root/tmva $Id$
// Authors: Simone Azeglio, Lorenzo Moneta , Stefan Wunsch

/*************************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis          *
 * Package: TMVA                                                                     *
 * Class  : RVariablePlotter                                                         *
 * Web    : http://tmva.sourceforge.net                                              *
 *                                                                                   *
 * Description:                                                                      *
 *      Variable Plotter                                                             *
 *                                                                                   *
 * Authors (alphabetical):                                                           *
 *      Simone Azeglio, University of Turin (Master Student), CERN (Summer Student)  *
 *      Lorenzo Moneta, CERN                                                         *
 *      Stefan Wunsch                                                                *
 *                                                                                   *
 * Copyright (c) 2021:                                                               *
 *                                                                                   *
 * Redistribution and use in source and binary forms, with or without                *
 * modification, are permitted according to the terms listed in LICENSE              *
 * (http://tmva.sourceforge.net/LICENSE)                                             *
 **********************************************************************************/


/*! \class TMVA::RVariablePlotter
\ingroup TMVA
Plotting a single variable
*/

#include "TMVA/RVariablePlotter.h"


////////////////////////////////////////////////////////////////////////////////
/// constructor with nodes (samples) and labels

TMVA::RVariablePlotter::RVariablePlotter( const std::vector<ROOT::RDF::RNode>& nodes, const std::vector<std::string>& labels)
    : fNodes(nodes),
    fLabels(labels){
        
        if (fNodes.size() != fLabels.size())
            std::runtime_error("Number of given RDataFrame nodes does not match number of given class labels.");

        if (fNodes.size() == 0)
            std::runtime_error("Number of given RDataFrame nodes and number of given class labels cannot be zero.");
}


////////////////////////////////////////////////////////////////////////////////
/// Drawing variables' plot

void TMVA::RVariablePlotter::Draw(const std::string& variable) {
   // Make histograms with TH1D
    const auto size = fNodes.size();
    std::vector<ROOT::RDF::RResultPtr<TH1D>> histos;
    
    for (std::size_t i = 0; i < size; i++) {
        // Trigger event loop with computing the histogram
    auto h = fNodes[i].Histo1D(variable);
        histos.push_back(h);
    }

   // Modify style and draw histograms
    THStack stack;
    
    for (unsigned int i = 0; i < histos.size(); i++) {
        histos[i]->SetLineColor(i + 1);
        
        /*if (i == 0) {
         histos[i]->SetTitle("");
         histos[i]->SetStats(false);
        }
        */
        histos[i]->SetTitle(variable.c_str());
        histos[i]->SetStats(true);
        stack.Add(histos[i].GetPtr());
   }
    
    auto clone = (THStack*) stack.DrawClone("nostack");
    
    clone->SetTitle(variable.c_str());
    clone->GetXaxis()->SetTitle(variable.c_str());
    clone->GetYaxis()->SetTitle("Count");
    
        
}


////////////////////////////////////////////////////////////////////////////////
/// Drawing Legend

void TMVA::RVariablePlotter::DrawLegend(float minX = 0.8, float minY = 0.8, float maxX = 0.9, float maxY = 0.9) {
    // make Legend from TLegend
    TLegend l(minX, minY, maxX, maxY);
    std::vector<TH1D> histos(fLabels.size());

    for (unsigned int i = 0; i < fLabels.size(); i++) {
        histos[i].SetLineColor(i + 1);
        l.AddEntry(&histos[i], fLabels[i].c_str(), "l");
    }
    l.SetBorderSize(1);
    l.DrawClone();
}
