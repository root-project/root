/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides an example of using a trained model with Keras
/// and make inference using SOFIE and RDataFrame
/// This macro uses as input a Keras model generated with the
/// TMVA_Higgs_Classification.C tutorial
/// You need to run that macro before this one.
/// In this case we are parsing the input file and then run the inference in the same
/// macro making use of the ROOT JITing capability
///
///
/// \macro_code
/// \macro_output
/// \author Lorenzo Moneta

using namespace TMVA::Experimental;

#include "TMVA/SOFIEHelpers.hxx"

void TMVA_SOFIE_RDataFrame_JIT(const char * modelFile = "Higgs_trained_model.h5"){

    TMVA::PyMethodBase::PyInitialize();

    // check if the input file exists
    TString modelFile = "Higgs_trained_model.h5";
    if (gSystem->AccessPathName(modelFile)) {
        Info("TMVA_SOFIE_RDataFrame","You need to run TMVA_Higgs_Classification.C to generate the Keras trained model");
        return;
    }

    // parse the input Keras model into RModel object
    SOFIE::RModel model = SOFIE::PyKeras::Parse(std::string(modelFile.Data()));

    modelFile.ReplaceAll(".h5",".hxx");
    //Generating inference code
    model.Generate();
    model.OutputGenerated(std::string(modelFile.Data()));
    model.PrintGenerated();

    // now compile using ROOT JIT trained model
    gROOT->ProcessLine(TString(".L ") + modelFile);


    std::string inputFileName = "Higgs_data.root";
    std::string inputFile = "http://root.cern.ch/files/" + inputFileName;

    gROOT->ProcessLine("auto sofie_functor = SofieFunctor<7,TMVA_SOFIE_Higgs_trained_model::Session>(0)");

    int nthreads = ROOT::GetThreadPoolSize();
    ROOT::RDataFrame df1("sig_tree", inputFile);
    auto h1 = df1.Define("DNN_Value", "sofie_functor(rdfslot_,m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)")
                   .Histo1D({"h_sig", "", 100, 0, 1},"DNN_Value");

    ROOT::RDataFrame df2("bkg_tree", inputFile);
    auto h2 = df2.Define("DNN_Value", "sofie_functor(rdfslot_,m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)")
                   .Histo1D({"h_bkg", "", 100, 0, 1},"DNN_Value");


    h1->SetLineColor(kRed);
    h2->SetLineColor(kBlue);

    auto c1 = new TCanvas();
    gStyle->SetOptStat(0);

    h2->DrawClone();
    h1->DrawClone("SAME");
    c1->BuildLegend();


}
