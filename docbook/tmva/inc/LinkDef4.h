#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link off all namespaces;

#pragma link C++ nestedclass;

#pragma link C++ namespace TMVA;

// other classes
#pragma link C++ class TMVA::TNeuron+;
#pragma link C++ class TMVA::TSynapse+;
#pragma link C++ class TMVA::TActivationChooser+;
#pragma link C++ class TMVA::TActivation+;
#pragma link C++ class TMVA::TActivationSigmoid+;
#pragma link C++ class TMVA::TActivationIdentity+;
#pragma link C++ class TMVA::TActivationTanh+;
#pragma link C++ class TMVA::TActivationRadial+;
#pragma link C++ class TMVA::TNeuronInputChooser+;
#pragma link C++ class TMVA::TNeuronInput+;
#pragma link C++ class TMVA::TNeuronInputSum+;
#pragma link C++ class TMVA::TNeuronInputSqSum+;
#pragma link C++ class TMVA::TNeuronInputAbs+;
#pragma link C++ class TMVA::Types+;
#pragma link C++ class TMVA::Ranking+;
#pragma link C++ class TMVA::RuleFit+;
#pragma link C++ class TMVA::RuleFitAPI+;
#pragma link C++ class TMVA::IMethod+;
#pragma link C++ class TMVA::MsgLogger+;
#pragma link C++ class TMVA::VariableTransformBase+;
#pragma link C++ class TMVA::VariableIdentityTransform+;
#pragma link C++ class TMVA::VariableDecorrTransform+;
#pragma link C++ class TMVA::VariablePCATransform+;
#pragma link C++ class TMVA::VariableGaussTransform+;
#pragma link C++ class TMVA::VariableNormalizeTransform+;

#endif
