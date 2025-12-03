// For RHist and RHistEngine, we request dictionaries for the most commonly used bin content types. This results in
// proper error messages when trying to stream. Other instantiations will be caught by the RAxes member.
#pragma link C++ class ROOT::Experimental::RHist<int>-;
#pragma link C++ class ROOT::Experimental::RHist<unsigned>-;
#pragma link C++ class ROOT::Experimental::RHist<long>-;
#pragma link C++ class ROOT::Experimental::RHist<unsigned long>-;
#pragma link C++ class ROOT::Experimental::RHist<long long>-;
#pragma link C++ class ROOT::Experimental::RHist<unsigned long long>-;
#pragma link C++ class ROOT::Experimental::RHist<float>-;
#pragma link C++ class ROOT::Experimental::RHist<double>-;
#pragma link C++ class ROOT::Experimental::RHist<ROOT::Experimental::RBinWithError>-;

#pragma link C++ class ROOT::Experimental::RHistEngine<int>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<unsigned>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<long>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<unsigned long>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<long long>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<unsigned long long>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<float>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<double>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<ROOT::Experimental::RBinWithError>-;

#pragma link C++ class ROOT::Experimental::RCategoricalAxis-;
#pragma link C++ class ROOT::Experimental::RHistStats-;
#pragma link C++ class ROOT::Experimental::RRegularAxis-;
#pragma link C++ class ROOT::Experimental::RVariableBinAxis-;
#pragma link C++ class ROOT::Experimental::Internal::RAxes-;
