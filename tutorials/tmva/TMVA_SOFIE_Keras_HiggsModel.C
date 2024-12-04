/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro run the SOFIE parser on the Keras model
/// obtaining running TMVA_Higgs_Classification.C
/// You need to run that macro before this one
///
/// \author Lorenzo Moneta

using namespace TMVA::Experimental;


void TMVA_SOFIE_Keras_HiggsModel(const char * modelFile = "Higgs_trained_model.h5"){

     // check if the input file exists
    if (gSystem->AccessPathName(modelFile)) {
        Error("TMVA_SOFIE_RDataFrame","You need to run TMVA_Higgs_Classification.C to generate the Keras trained model");
        return;
    }

    // parse the input Keras model into RModel object
    SOFIE::RModel model = SOFIE::PyKeras::Parse(modelFile);

    TString modelHeaderFile = modelFile;
    modelHeaderFile.ReplaceAll(".h5",".hxx");
    //Generating inference code
    model.Generate();
    model.OutputGenerated(std::string(modelHeaderFile));

    // copy include in $ROOTSYS/tutorials/
    std::cout << "include is in " << gROOT->GetIncludeDir() << std::endl;
}
