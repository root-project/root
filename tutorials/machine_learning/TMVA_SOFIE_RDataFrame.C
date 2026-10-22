/// \file
/// \ingroup tutorial_ml
/// \notebook -nodraw
/// This macro provides an example of using a trained model with PyTorch
/// and make inference using SOFIE and RDataFrame
/// This macro uses as input an ONNX model generated with the
/// Python tutorial TMVA_SOFIE_PyTorch_HiggsModel.py
/// You need to run that macro before to generate the trained PyTorch model
/// and also the  corresponding header file with SOFIE which can then be used for inference
///
/// Execute in this order:
/// ```
/// python3 TMVA_SOFIE_PyTorch_HiggsModel.py
/// root TMVA_SOFIE_RDataFrame.C
/// ```
///
/// \macro_code
/// \macro_output
/// \author Lorenzo Moneta

// need to add the current directory (from where we are running this macro)
// to the include path for Cling
R__ADD_INCLUDE_PATH($PWD)
#include "HiggsModel.hxx"

#include <array>
#include <vector>

void TMVA_SOFIE_RDataFrame(int nthreads = 2){

   std::string inputFileName = "Higgs_data.root";
   std::string inputFile = std::string{gROOT->GetTutorialDir()} + "/machine_learning/data/" + inputFileName;

   ROOT::EnableImplicitMT(nthreads);

   ROOT::RDataFrame df1("sig_tree", inputFile);
   int nslots = df1.GetNSlots();
   std::cout << "Running using " << nslots << " threads" << std::endl;

   // A SOFIE Session holds the model weights and the intermediate buffers and is
   // not thread-safe: create one Session per RDataFrame processing slot and use
   // the slot number in the DefineSlot functor to dispatch to the right one.
   // The Session default constructor reads the weights from the default weight
   // file (HiggsModel.dat in this case).
   std::vector<TMVA_SOFIE_HiggsModel::Session> sessions(nslots);

   // The functor assembles the model input tensor from the RDataFrame columns
   // and evaluates the model. The column order must match the ordering of the
   // model input tensor.
   auto evalModel = [&sessions](unsigned int slot, float m_jj, float m_jjj, float m_lv, float m_jlv, float m_bb,
                                float m_wbb, float m_wwbb) {
      std::array<float, 7> input{m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb};
      auto result = sessions[slot].infer(input.data());
      return result[0];
   };

   auto h1 = df1.DefineSlot("DNN_Value", evalModel, {"m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"})
                .Histo1D({"h_sig", "", 100, 0, 1}, "DNN_Value");

   ROOT::RDataFrame df2("bkg_tree", inputFile);
   auto h2 = df2.DefineSlot("DNN_Value", evalModel, {"m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"})
                .Histo1D({"h_bkg", "", 100, 0, 1}, "DNN_Value");

   h1->SetLineColor(kRed);
   h2->SetLineColor(kBlue);

   auto c1 = new TCanvas();
   gStyle->SetOptStat(0);

   h2->DrawClone();
   h1->DrawClone("SAME");
   c1->BuildLegend();

}
