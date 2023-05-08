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

/// function to compile the generated model and the declaration of the SofieFunctor
/// used by RDF.
/// Assume that the model name as in the header file
void CompileModelForRDF(const std::string & headerModelFile, unsigned int ninputs, unsigned int nslots=0) {

   std::string modelName = headerModelFile.substr(0,headerModelFile.find(".hxx"));
   std::string cmd = std::string("#include \"") + headerModelFile + std::string("\"");
   auto ret = gInterpreter->Declare(cmd.c_str());
   if (!ret)
    throw std::runtime_error("Error compiling : " + cmd);
   std::cout << "compiled : " << cmd << std::endl;

   cmd = "auto sofie_functor = TMVA::Experimental::SofieFunctor<" + std::to_string(ninputs) + ",TMVA_SOFIE_" +
    modelName + "::Session>(" + std::to_string(nslots) + ");";
   ret = gInterpreter->Declare(cmd.c_str());
   if (!ret)
    throw std::runtime_error("Error compiling : " + cmd);
   std::cout << "compiled : " << cmd << std::endl;
   std::cout << "Model is ready to be evaluated" << std::endl;
   return;
}

void TMVA_SOFIE_RDataFrame_JIT(std::string modelFile = "Higgs_trained_model.h5"){

    TMVA::PyMethodBase::PyInitialize();

    // check if the input file exists
    if (gSystem->AccessPathName(modelFile.c_str())) {
        Info("TMVA_SOFIE_RDataFrame","You need to run TMVA_Higgs_Classification.C to generate the Keras trained model");
        return;
    }

    // parse the input Keras model into RModel object
    SOFIE::RModel model = SOFIE::PyKeras::Parse(modelFile);

    std::string modelName = modelFile.substr(0,modelFile.find(".h5"));
    std::string modelHeaderFile = modelName + std::string(".hxx");
    //Generating inference code
    model.Generate();
    model.OutputGenerated(modelHeaderFile);
    model.PrintGenerated();
    // check that also weigh file exists
    std::string modelWeightFile = modelName + std::string(".dat");
    if (gSystem->AccessPathName(modelWeightFile.c_str())) {
        Error("TMVA_SOFIE_RDataFrame","Generated weight file is missing");
        return;
    }

    // now compile using ROOT JIT trained model (see function above)
    CompileModelForRDF(modelHeaderFile,7);

    std::string inputFileName = "Higgs_data.root";
    std::string inputFile = "http://root.cern.ch/files/" + inputFileName;

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
