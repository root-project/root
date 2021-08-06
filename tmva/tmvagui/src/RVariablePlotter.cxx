// @(#)root/tmva $Id$
// Authors: Simone Azeglio, Sitong An, Lorenzo Moneta , Stefan Wunsch

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
 *      Lorenzo Moneta, CERN
 *      Sitong An, CERN / CMU                                                        *
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
#include "TMVA/tmvaglob.h"
#include "TMVA/RTensor.hxx"
#include "TMVA/ROCCurve.h"


////////////////////////////////////////////////////////////////////////////////
/// constructor for RDataframe with nodes (samples) and labels

TMVA::RVariablePlotter::RVariablePlotter(const std::vector<ROOT::RDF::RNode>& nodes, const std::vector<std::string>& labels)
    : fNodes(nodes),
    fLabels(labels){
        
        if (fNodes.size() != fLabels.size())
            std::runtime_error("Number of given RDataFrame nodes does not match number of given class labels.");

        if (fNodes.size() == 0)
            std::runtime_error("Number of given RDataFrame nodes and number of given class labels cannot be zero.");
}


////////////////////////////////////////////////////////////////////////////////
/// constructor (overloading) for RTensor nodes (samples) and labels

TMVA::RVariablePlotter::RVariablePlotter( const std::vector<TMVA::Experimental::RTensor<Float_t>>& tensors, const std::vector<std::string>& labels)
    : fTensors(tensors),
    fLabels(labels)
    {
        
        if (fTensors.size() != fLabels.size())
            std::runtime_error("Number of given RTensor components does not match number of given class labels.");

        if (fTensors.size() == 0)
            std::runtime_error("Number of given RTensor components and number of given class labels cannot be zero.");
}

////////////////////////////////////////////////////////////////////////////////
/// Set style and keep existing canvas
void TMVA::RVariablePlotter::InitializeStyle(bool useTMVAStyle){
   // set custom style
   if (!useTMVAStyle) {
      gROOT->SetStyle("Plain");
      gStyle->SetOptStat(0);
      gPad->SetMargin(0.2, 0.9, 0.1, 0.9);
      gPad->SetGrid(1,1);
      return;
   }
    
   TMVA::TMVAGlob::SetTMVAStyle();
   gPad->SetMargin(0.2, 0.9, 0.1, 0.9);
}

////////////////////////////////////////////////////////////////
/// Drawing variables' plot RDataframe
void TMVA::RVariablePlotter::Draw(const std::string& variable, bool useTMVAStyle) {
   // Make histograms with TH1D
    TMVA::RVariablePlotter::InitializeStyle(useTMVAStyle);
    
    const auto size = fNodes.size();
    std::vector<ROOT::RDF::RResultPtr<TH1D>> histos;
    
    for (std::size_t i = 0; i < size; i++) {
        // Trigger event loop with computing the histogram
        ROOT::RDF::RResultPtr<TH1D> h = fNodes[i].Histo1D(variable);
        histos.push_back(h);
    }

   // Modify style and draw histograms
    THStack stack_;
    
    for (unsigned int i = 0; i < histos.size(); i++) {
        histos[i]->SetLineColor(i + 1);
        histos[i]->SetTitle(variable.c_str());
        histos[i]->SetStats(true);
        stack_.Add(histos[i].GetPtr());
   }
    
    auto clone = (THStack*) stack_.DrawClone("nostack");
    
    clone->SetTitle(variable.c_str());
    clone->GetXaxis()->SetTitle(variable.c_str());
    clone->GetYaxis()->SetTitle("Count");
    clone->GetYaxis()->SetTitleOffset( 1.50 );
        
}

////////////////////////////////////////////////////////////////
/// Drawing variables' plot from RTensor input
void TMVA::RVariablePlotter::DrawTensor(const std::string& variable, const std::vector<std::string>& variables, bool useTMVAStyle) {
   
    // vector of RNodes
    auto RNodeVec = TMVA::RVariablePlotter::TensorsToNodes(variables);
    
    TMVA::RVariablePlotter::InitializeStyle(useTMVAStyle);
    
    // Make histograms with TH1D
    const auto size = RNodeVec.size();
    std::vector<ROOT::RDF::RResultPtr<TH1D>> histos;
    
    for (std::size_t i = 0; i < size; i++) {
        // Trigger event loop with computing the histogram
        ROOT::RDF::RResultPtr<TH1D> h = RNodeVec[i].Histo1D(variable);
        histos.push_back(h);
        h = ROOT::RDF::RResultPtr<TH1D>();
    }

   // Modify style and draw histograms
    THStack stack;
    
    for (unsigned int i = 0; i < histos.size(); i++) {
        histos[i]->SetLineColor(i + 1);
        histos[i]->SetTitle(variable.c_str());
        histos[i]->SetStats(true);
        stack.Add(histos[i].GetPtr());
   }
    
    auto clone = (THStack*) stack.DrawClone("nostack");
    
    clone->SetTitle(variable.c_str());
    clone->GetXaxis()->SetTitle(variable.c_str());
    clone->GetYaxis()->SetTitle("Count");
    clone->GetYaxis()->SetTitleOffset( 1.50 );
        
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

////////////////////////////////////////////////////////////////////////////////
/// Drawing ROC Curve - RNode input, multiple models
void TMVA::RVariablePlotter::DrawMultiROCCurve(const std::vector<std::string>& output_variables, bool useTMVAStyle){

    // consistency check, i.e. we need signal vs background for the roc curve
    if (fNodes.size() != 2)
        std::runtime_error("In order to plot the ROC Curve you need 2 RNodes, corresponding to prediction on signal and background");
    
    // TMVA plotting style
    TMVA::RVariablePlotter::InitializeStyle(useTMVAStyle);

    auto sigDf = fNodes[0]; 
    auto bkgDf = fNodes[1];

    // here we can store signals and backgrounds from each model
    std::vector<std::vector<float>> sigModels; 
    std::vector<std::vector<float>> bkgModels; 

    for (unsigned int i = 0; i < output_variables.size(); i++){
        // extract columns as vectors of float for the selected model (i.e. specified by output_variable)
        sigModels.push_back(sigDf.Take<float>(output_variables[i]).GetValue());
        bkgModels.push_back(bkgDf.Take<float>(output_variables[i]).GetValue());
    }

   std::string title = "ROC Curves";

    // ROC first curve 
    auto ROC = TMVA::ROCCurve(sigModels[0], bkgModels[0]); 
    // Label = AUC Score for each model as sequence of char
    std::string AUC = std::to_string(ROC.GetROCIntegral()); 
    std::string legend = " AUC = "; 
    std::string legendStr = legend.append(AUC); 
    std::string out_var = output_variables[0];
    std::string labelName = out_var.append(legendStr); 
    const char *ROCAUCLabel = labelName.c_str();

    auto clone = (TGraph*) ROC.GetROCCurve(); 
    clone->SetTitle(title.c_str());
    clone->SetLineColor(1); 
    // axis of the ROC Curve
    clone->GetXaxis()->SetTitle("Specificity"); 
    clone->GetYaxis()->SetTitle("Sensititvy");
    clone->GetYaxis()->SetTitleOffset(1.0);
    clone->DrawClone("AC");  

    // Legend
    auto lROC = new TLegend(0.65, 0.75, 0.90, 0.90);
    lROC->AddEntry(clone, ROCAUCLabel, "l");
    lROC->DrawClone(); 

    for (unsigned int i = 1; i < sigModels.size(); i++){
        // ROC instance
        auto ROC = TMVA::ROCCurve(sigModels[i], bkgModels[i]); 
        // Label = AUC Score for each model as sequence of char
        std::string AUC = std::to_string(ROC.GetROCIntegral()); 
        std::string legend = " AUC = "; 
        std::string legendStr = legend.append(AUC); 
        std::string out_var = output_variables[i];
        std::string labelName = out_var.append(legendStr); 
        const char *ROCAUCLabel = labelName.c_str();

        auto clone = (TGraph*) ROC.GetROCCurve(); 
        clone->SetLineColor(i + 1); 
        clone->DrawClone("CP");  
        lROC->AddEntry(clone, ROCAUCLabel, "l");
        lROC->DrawClone();
    }    

}

////////////////////////////////////////////////////////////////////////////////
/// Drawing ROC Curve - RTensor input, multiple models, toDo waiting for legend bug to be solved

////////////////////////////////////////////////////////////////////////////////
/// Drawing ROC Curve - RNode input, single model 
void TMVA::RVariablePlotter::DrawROCCurve(const std::string& output_variable , bool useTMVAStyle){

    // consistency check, i.e. we need signal vs background for the roc curve
    if (fNodes.size() != 2)
        std::runtime_error("In order to plot the ROC Curve you need 2 RNodes, corresponding to prediction on signal and background");
    
    // TMVA plotting style
    TMVA::RVariablePlotter::InitializeStyle(useTMVAStyle);

    auto sigDf = fNodes[0]; 
    auto bkgDf = fNodes[1];

    // extract columns as vectors of float for the selected model (i.e. specified by output_variable)
    auto sigModel = sigDf.Take<float>(output_variable).GetValue();
    auto bkgModel = bkgDf.Take<float>(output_variable).GetValue();

    // ROC Curve     
    auto ROC = TMVA::ROCCurve(sigModel, bkgModel); 
    auto ROCCurve = ROC.GetROCCurve(); 

    // ROC AUC Score as Legend's Label
    std::string AUC = std::to_string(ROC.GetROCIntegral()); 
    std::string legend = "AUC = "; 
    const std::string legendStr = legend.append(AUC); 
    const char *ROCAUCLabel = legendStr.c_str(); 

    // Graph title 
    std::string varName = output_variable; 
    std::string title = varName.append(" ROC Curve");

    auto clone = (TGraph*) ROCCurve; 
    
    clone->SetTitle(title.c_str());
    clone->SetLineColor(1); 
    // axis of the ROC Curve
    clone->GetXaxis()->SetTitle("Specificity"); 
    clone->GetYaxis()->SetTitle("Sensititvy");
    clone->GetYaxis()->SetTitleOffset(1.0);
    clone->DrawClone("AC"); 

    TLegend lROC(0.5, 0.8, 0.95, 0.9);
    lROC.AddEntry("clone", ROCAUCLabel, "lROC"); 
    lROC.DrawClone(); 

}

////////////////////////////////////////////////////////////////////////////////
/// Drawing ROC Curve - RTensor input, single model 
void TMVA::RVariablePlotter::DrawROCCurveTensor(const std::string& output_variable, const std::vector<std::string>& output_variables, bool useTMVAStyle){

    // consistency check, i.e. we need signal vs background for the roc curve
    if (fTensors.size() != 2)
        std::runtime_error("In order to plot the ROC Curve you need 2 RTensors, corresponding to prediction on signal and background");
    
    auto RNodeVec = TensorsToNodes(output_variables);

    // TMVA plotting style
    TMVA::RVariablePlotter::InitializeStyle(useTMVAStyle);

    auto sigDf = RNodeVec[0]; 
    auto bkgDf = RNodeVec[1];

    // extract columns as vectors of float for the selected model (i.e. specified by output_variable)
    auto sigModel = sigDf.Take<float>(output_variable).GetValue();
    auto bkgModel = bkgDf.Take<float>(output_variable).GetValue();

    // ROC Curve     
    auto ROC = TMVA::ROCCurve(sigModel, bkgModel); 
    auto ROCCurve = ROC.GetROCCurve(); 

    // ROC AUC Score as Legend's Label
    std::string AUC = std::to_string(ROC.GetROCIntegral()); 
    std::string legend = "AUC = "; 
    const std::string legendStr = legend.append(AUC); 
    const char *ROCAUCLabel = legendStr.c_str(); 

    // Graph title 
    std::string varName = output_variable; 
    std::string title = varName.append(" ROC Curve");

    auto clone = (TGraph*) ROCCurve; 
    
    clone->SetTitle(title.c_str());
    clone->SetLineColor(1); 
    // axis of the ROC Curve
    clone->GetXaxis()->SetTitle("Specificity"); 
    clone->GetYaxis()->SetTitle("Sensititvy");
    clone->GetYaxis()->SetTitleOffset(1.0);
    clone-> DrawClone("AC"); 

    TLegend lROC(0.5, 0.8, 0.95, 0.9);
    lROC.AddEntry("clone", ROCAUCLabel, "lROC"); 
    lROC.DrawClone();

}

////////////////////////////////////////////////////////////////////////////////
/// RTensor to RNode Converter (Static Function - Private)

// add variables argument for custom variables
ROOT::RDF::RNode TMVA::RVariablePlotter::TensorToNode(const TMVA::Experimental::RTensor<Float_t>& tensor,  const std::vector<std::string>& variables){
        
    // shape check
    if (tensor.GetShape().size() != 2)
        std::runtime_error("Number of given RTensor dimensions does not match number of given class labels.");
    
    auto dfSig = ROOT::RDataFrame(tensor.GetShape()[0]).DefineSlotEntry(variables[0], [=] (unsigned, ULong64_t entry) { return tensor(entry, 0); } );
    
    std::size_t nvar = variables.size();
    // in case RTensors have a different number of columns
    if(tensor.GetShape()[1]!= variables.size())
        nvar = std::min(tensor.GetShape()[1], variables.size());
    
    for (std::size_t j = 1; j < nvar; j++){
        dfSig = dfSig.DefineSlotEntry(variables[j],  [=] (unsigned, ULong64_t entry) { return tensor(entry,j);});
            }
    
    // RDataFrame to RNode
    auto DFNode = ROOT::RDF::RNode(dfSig);
    
    return DFNode;
}

////////////////////////////////////////////////////////////////////////////////
/// RTensor vector to RNode vector converter (Private)
std::vector<ROOT::RDF::RNode> TMVA::RVariablePlotter::TensorsToNodes(const std::vector<std::string>& variables){
    
    const auto size = fTensors.size();
    std::vector<ROOT::RDF::RNode> RNodeVec;

    // loop through tensors
    for (std::size_t k = 0; k < size; k++){  
        RNodeVec.push_back(TensorToNode(fTensors[k], variables));
    }
    
    return RNodeVec;
}


