#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class MyObject+;
#pragma link C++ class MemberMyObject+;
#pragma link C++ class vector<int>+; // removed the '+'

#endif
