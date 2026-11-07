/// \file
/// \ingroup tutorial_ml
/// \notebook -nodraw
/// This macro provides an example of using a trained model with PyTorch
/// and make inference using SOFIE and RDataFrame
/// This macro uses as input the SOFIE header generated from the ONNX model
/// with the TMVA_SOFIE_PyTorch_HiggsModel.py tutorial
/// You need to run that macro before this one.
/// In this case we are parsing the input file and then run the inference in the same
/// macro making use of the ROOT JITing capability
///
///
/// \macro_code
/// \macro_output
/// \author Lorenzo Moneta

/// Function to compile the generated model with the ROOT JIT and to declare the
/// Session objects and the inference function used by RDataFrame.
/// A SOFIE Session holds the model weights and the intermediate buffers and is
/// not thread-safe: one Session per RDataFrame processing slot is created and
/// the slot number is used to dispatch to the right one.
/// Assume that the model name is the same as the header file name.
void CompileModelForRDF(const std::string &headerModelFile, unsigned int ninputs, unsigned int nslots = 0)
{

   std::string modelName = headerModelFile.substr(0,headerModelFile.find(".hxx"));
   std::string cmd =
      std::string("#include \"") + headerModelFile + std::string("\"\n#include <array>\n#include <vector>");
   auto ret = gInterpreter->Declare(cmd.c_str());
   if (!ret)
    throw std::runtime_error("Error compiling : " + cmd);
   std::cout << "compiled : " << cmd << std::endl;

   // Declare one Session per processing slot. The Session default constructor
   // reads the weights from the default weight file (<modelName>.dat here).
   if (nslots < 1)
      nslots = 1;
   cmd = "std::vector<TMVA_SOFIE_" + modelName + "::Session> sofie_sessions(" + std::to_string(nslots) + ");";
   ret = gInterpreter->Declare(cmd.c_str());
   if (!ret)
      throw std::runtime_error("Error compiling : " + cmd);

   // Declare the inference function for RDataFrame: it assembles the model
   // input tensor from the columns and evaluates the model of the given slot.
   std::string params;
   std::string inputValues;
   for (unsigned int i = 0; i < ninputs; i++) {
      if (i > 0) {
         params += ", ";
         inputValues += ", ";
      }
      params += "float x" + std::to_string(i);
      inputValues += "x" + std::to_string(i);
   }
   cmd = "double sofie_eval(unsigned int slot, " + params +
         ") {\n"
         "   std::array<float, " +
         std::to_string(ninputs) + "> input{" + inputValues +
         "};\n"
         "   return sofie_sessions[slot].infer(input.data())[0];\n"
         "}";
   ret = gInterpreter->Declare(cmd.c_str());
   if (!ret)
    throw std::runtime_error("Error compiling : " + cmd);
   std::cout << "compiled : " << cmd << std::endl;
   std::cout << "Model is ready to be evaluated" << std::endl;
   return;
}

void TMVA_SOFIE_RDataFrame_JIT(std::string modelName = "HiggsModel"){

    // check if the input file exists
    std::string modelHeaderFile = modelName + ".hxx";
    if (gSystem->AccessPathName(modelHeaderFile.c_str())) {
       Info("TMVA_SOFIE_RDataFrame", "You need to run TMVA_SOFIE_PyTorch_HiggsModel.py to generate the SOFIE header "
                                     "for the PyTorch trained model");
       return;
    }

    // check that also weigh file exists
    std::string modelWeightFile = modelName + std::string(".dat");
    if (gSystem->AccessPathName(modelWeightFile.c_str())) {
        Error("TMVA_SOFIE_RDataFrame","Generated weight file is missing");
        return;
    }

    // now compile using ROOT JIT trained model (see function above)
    CompileModelForRDF(modelHeaderFile,7);

    std::string inputFileName = "Higgs_data.root";
    std::string inputFile = std::string{gROOT->GetTutorialDir()} + "/machine_learning/data/" + inputFileName;

    // The column order in the Define expressions must match the ordering of the
    // model input tensor.
    ROOT::RDataFrame df1("sig_tree", inputFile);
    auto h1 = df1.Define("DNN_Value", "sofie_eval(rdfslot_,m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)")
                 .Histo1D({"h_sig", "", 100, 0, 1}, "DNN_Value");

    ROOT::RDataFrame df2("bkg_tree", inputFile);
    auto h2 = df2.Define("DNN_Value", "sofie_eval(rdfslot_,m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)")
                 .Histo1D({"h_bkg", "", 100, 0, 1}, "DNN_Value");

    h1->SetLineColor(kRed);
    h2->SetLineColor(kBlue);

    auto c1 = new TCanvas();
    gStyle->SetOptStat(0);

    h2->DrawClone();
    h1->DrawClone("SAME");
    c1->BuildLegend();


}
