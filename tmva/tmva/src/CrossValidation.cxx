// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson, Pourya Vakilipourtakalou.

#include <iostream>

#include "TMVA/CrossValidation.h"
#include "TMVA/MethodBase.h"
#include "TMVA/ResultsClassification.h"
#include "TSystem.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"
#include <memory>
#include "TMVA/tmvaglob.h"

TMVA::CrossValidationResult::CrossValidationResult():fROCCurves(new TMultiGraph())
{
}

TMVA::CrossValidationResult::CrossValidationResult(const CrossValidationResult &obj)
{
    fROCs=obj.fROCs;
    fROCCurves = obj.fROCCurves;
}

TMVA::CrossValidationResult::~CrossValidationResult()
{
    fROCCurves=nullptr;
}

std::shared_ptr<TMultiGraph> &TMVA::CrossValidationResult::GetROCCurves()
{
    return fROCCurves;
}

void TMVA::CrossValidationResult::SetROCValue(UInt_t fold,Float_t rocint)
{
    fROCs[fold]=rocint;
}


Float_t TMVA::CrossValidationResult::GetROCAverage() const
{
    Float_t avg=0;
    for(auto &roc:fROCs) avg+=roc.second;
    return avg/fROCs.size();
}


void TMVA::CrossValidationResult::Print() const
{    
    MsgLogger fLogger("CrossValidation");
    for(auto &item:fROCs)
        fLogger<<kINFO<<Form("Fold  %i ROC-Int : %f",item.first,item.second)<<std::endl;
    
    fLogger<<kINFO<<Form("Average ROC-Int : %f",GetROCAverage())<<Endl;

}


TCanvas* TMVA::CrossValidationResult::Draw(const TString name) const
{
    TCanvas *c=new TCanvas(name.Data());
    fROCCurves->Draw("AL");
    fROCCurves->GetXaxis()->SetTitle(" Signal Efficiency ");
    fROCCurves->GetYaxis()->SetTitle(" Background Rejection ");
    Float_t adjust=1+fROCs.size()*0.01;
    c->BuildLegend(0.15,0.15,0.4*adjust,0.5*adjust);
    c->SetTitle("Cross Validation ROC Curves");
    c->Draw();
    return c;
}
