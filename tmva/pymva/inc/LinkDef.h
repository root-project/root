#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link off all namespaces;

#pragma link C++ nestedclass;


// the classifiers
#pragma link C++ class TMVA::PyMethodBase+;
#pragma link C++ class TMVA::MethodPyRandomForest+;
#pragma link C++ class TMVA::MethodPyAdaBoost+;
#pragma link C++ class TMVA::MethodPyGTB+;
#pragma link C++ class TMVA::MethodPyKeras+;
#pragma link C++ class TMVA::MethodPyTorch+;
#pragma link C++ namespace TMVA::Experimental::SOFIE::PyKeras;
#pragma link C++ function TMVA::Experimental::SOFIE::PyKeras::Parse+;
#pragma link C++ namespace TMVA::Experimental::SOFIE::PyTorch;
#pragma link C++ function TMVA::Experimental::SOFIE::PyTorch::Parse+;
#endif
