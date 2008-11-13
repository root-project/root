#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ options=version(2) class ClassA+;
#pragma link C++ options=version(2) class ClassB+;
#pragma link C++ options=version(2) class ClassC+;
#pragma link C++ options=version(2) class ClassAIns+;
#pragma link C++ options=version(2) class ClassABase+;
#pragma link C++ class std::vector<ClassA>+;
#pragma link C++ class std::vector<std::pair<int,double> >+;
#pragma link C++ class std::vector<ClassA*>+;

#endif
