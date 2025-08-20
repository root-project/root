// For RHistEngine, we request dictionaries for the most commonly used bin content types. This results in proper error
// messages when trying to stream. Other instantiations will be caught by the RAxes member.
#pragma link C++ class ROOT::Experimental::RHistEngine<int>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<long>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<long long>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<float>-;
#pragma link C++ class ROOT::Experimental::RHistEngine<double>-;

#pragma link C++ class ROOT::Experimental::RRegularAxis-;
#pragma link C++ class ROOT::Experimental::RVariableBinAxis-;
#pragma link C++ class ROOT::Experimental::Internal::RAxes-;
