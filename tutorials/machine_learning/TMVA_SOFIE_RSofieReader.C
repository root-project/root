/// \file
/// \ingroup tutorial_ml
/// \notebook -nodraw
/// This macro provides an example of using a trained model with Keras
/// and make inference using SOFIE with the RSofieReader class
/// This macro uses as input a Keras model generated with the
/// TMVA_Higgs_Classification.C tutorial
/// You need to run that macro before to generate the trained Keras model
///
///
/// Execute in this order:
/// ```
/// root TMVA_Higgs_Classification.C
/// root TMVA_SOFIE_RSofieReader.C
/// ```
///
/// \macro_code
/// \macro_output
/// \author Lorenzo Moneta

using namespace TMVA::Experimental;

void TMVA_SOFIE_RSofieReader(){

   RSofieReader model("Higgs_trained_model.h5");
   // for debugging
   //RSofieReader model("Higgs_trained_model.h5", {}, true);

   // the input shape for this model is a tensor with shape (1,7)

   std::vector<float> input = {0.1,0.2,0.3,0.4,0.5,0.6,0.7};

   // predict model on a single event (takes a std::vector<float>)

   auto output = model.Compute(input);

   std::cout << "Event prediction = " << output[0] << std::endl;

   // predict model now on a input file using RDataFrame

   std::string inputFileName = "Higgs_data.root";
   std::string inputFile = gROOT->GetTutorialDir() + "/machine_learning/data/" + inputFileName;


   ROOT::RDataFrame df1("sig_tree", inputFile);

   auto h1 = df1.Define("DNN_Values", Compute<7, float>(model),
                            {"m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"})
                  .Define("y","DNN_Values[0]")
                  .Histo1D({"h_sig", "", 100, 0, 1}, "y");

   ROOT::RDataFrame df2("bkg_tree", inputFile);
   auto h2 = df2.Define("DNN_Values", Compute<7, float>(model),
                            {"m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"})
                  .Define("y","DNN_Values[0]")
                  .Histo1D({"h_bkg", "", 100, 0, 1}, "y");

   h1->SetLineColor(kRed);
   h2->SetLineColor(kBlue);

   auto c1 = new TCanvas();
   gStyle->SetOptStat(0);

   h2->DrawClone();
   h1->DrawClone("SAME");
   c1->BuildLegend();

}
