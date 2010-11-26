#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link off all namespaces;

#pragma link C++ nestedclass;

#pragma link C++ namespace TMVA;

// other classes
#pragma link C++ class TMVA::Config+;
#pragma link C++ class TMVA::Config::VariablePlotting+;
#pragma link C++ class TMVA::Config::IONames+;
#pragma link C++ class TMVA::KDEKernel+;
#pragma link C++ class TMVA::Interval+;
#pragma link C++ class TMVA::FitterBase+;
#pragma link C++ class TMVA::MCFitter+;
#pragma link C++ class TMVA::GeneticFitter+;
#pragma link C++ class TMVA::SimulatedAnnealingFitter+;
#pragma link C++ class TMVA::MinuitFitter+;
#pragma link C++ class TMVA::MinuitWrapper+;
#pragma link C++ class TMVA::IFitterTarget+;
#pragma link C++ class TMVA::PDEFoam+;
#pragma link C++ class TMVA::PDEFoamDistr+;
#pragma link C++ class TMVA::PDEFoamVect+;
#pragma link C++ class TMVA::PDEFoamCell+;
#pragma link C++ class TMVA::BDTEventWrapper+;
#pragma link C++ class TMVA::CCTreeWrapper+;
#pragma link C++ class TMVA::CCPruner+;
#pragma link C++ class TMVA::CostComplexityPruneTool+;
#pragma link C++ class TMVA::SVEvent+;
#pragma link C++ class TMVA::OptimizeConfigParameters+;

#endif
