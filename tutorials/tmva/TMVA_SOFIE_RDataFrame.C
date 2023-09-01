/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides an example of using a trained model with Keras
/// and make inference using SOFIE and RDataFrame
/// This macro uses as input a Keras model generated with the
/// TMVA_Higgs_Classification.C tutorial
/// You need to run that macro before to generate the trained Keras model
/// Then you need to run the macro TMVA_SOFIE_Keras_HiggsModel.C to generate the  corresponding
/// header file using SOFIE.
///
/// Execute in this order:
/// ```
/// root TMVA_Higgs_Classification.C
/// root TMVA_SOFIE_Keras_HiggsModel.C
/// root TMVA_SOFIE_RDataFrame.C
/// ```
///
/// \macro_code
/// \macro_output
/// \author Lorenzo Moneta

using namespace TMVA::Experimental;

// need to add the current directory (from where we are running this macro)
// to the include path for Cling
R__ADD_INCLUDE_PATH($PWD)
R__ADD_INCLUDE_PATH($ROOTSYS/runtutorials)
#include "Higgs_trained_model.hxx"
#include "TMVA/SOFIEHelpers.hxx"

using namespace TMVA::Experimental;

void TMVA_SOFIE_RDataFrame(int nthreads = 2){

   std::string inputFileName = "Higgs_data.root";
   std::string inputFile = "http://root.cern.ch/files/" + inputFileName;

   ROOT::EnableImplicitMT(nthreads);

   ROOT::RDataFrame df1("sig_tree", inputFile);
   int nslots = df1.GetNSlots();
   std::cout << "Running using " << nslots << " threads" << std::endl;
   auto h1 = df1.DefineSlot("DNN_Value", SofieFunctor<7, TMVA_SOFIE_Higgs_trained_model::Session>(nslots),
                            {"m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"})
                .Histo1D({"h_sig", "", 100, 0, 1}, "DNN_Value");

   ROOT::RDataFrame df2("bkg_tree", inputFile);
   nslots = df2.GetNSlots();
   auto h2 = df2.DefineSlot("DNN_Value", SofieFunctor<7, TMVA_SOFIE_Higgs_trained_model::Session>(nslots),
                            {"m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"})
                .Histo1D({"h_bkg", "", 100, 0, 1}, "DNN_Value");

   h1->SetLineColor(kRed);
   h2->SetLineColor(kBlue);

   auto c1 = new TCanvas();
   gStyle->SetOptStat(0);

   h2->DrawClone();
   h1->DrawClone("SAME");
   c1->BuildLegend();

}
