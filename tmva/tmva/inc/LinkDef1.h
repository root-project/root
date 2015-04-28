#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link off all namespaces;

#pragma link C++ nestedclass;


#pragma link C++ class TMVA::Configurable+;
#pragma link C++ class TMVA::Event+;
#pragma link C++ class TMVA::kNN::Event+;
#pragma link C++ class TMVA::Factory+;

// the classifiers
#pragma link C++ class TMVA::MethodBase+;
#pragma link C++ class TMVA::MethodCompositeBase+;
#pragma link C++ class TMVA::MethodANNBase+;
#pragma link C++ class TMVA::MethodTMlpANN+;
#pragma link C++ class TMVA::MethodRuleFit+;
#pragma link C++ class TMVA::MethodCuts+;
#pragma link C++ class TMVA::MethodFisher+;
#pragma link C++ class TMVA::MethodKNN+;
#pragma link C++ class TMVA::MethodCFMlpANN+;
#pragma link C++ class TMVA::MethodCFMlpANN_Utils+;
#pragma link C++ class TMVA::MethodLikelihood+;
#pragma link C++ class TMVA::MethodHMatrix+;
#pragma link C++ class TMVA::MethodPDERS+;
#pragma link C++ class TMVA::MethodBDT+;
#pragma link C++ class TMVA::MethodDT+;
#pragma link C++ class TMVA::MethodSVM+;
#pragma link C++ class TMVA::MethodBayesClassifier+;
#pragma link C++ class TMVA::MethodFDA+;
#pragma link C++ class TMVA::MethodMLP+;
#pragma link C++ class TMVA::MethodBoost+;
#pragma link C++ class TMVA::MethodPDEFoam+;
#pragma link C++ class TMVA::MethodLD+;
#pragma link C++ class TMVA::MethodCategory+;

#endif
