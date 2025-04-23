#ifdef __CINT__

#pragma link off all globals;
#pragma link off all functions;
#pragma link off all classes;
#pragma link off all namespaces;

#pragma link C++ class MyClass+;
#pragma link C++ function MyClass::GetScalar<Int_t>(TString);

#pragma link C++ function GetMyClassReference();

#endif
