/*
 Test conversion operators.
 #define CINTFAILURE to see all failures */

// #define CINTFAILURE

void runConversionOp() {
   gSystem->Setenv("LINES","-1");

   // Make sure the library is not loaded instead of 
   // the script.
   gInterpreter->UnloadLibraryMap("ConversionOp_h");
   gInterpreter->UnloadLibraryMap("equal_C");
   
   gROOT->ProcessLine(".x ConversionOp.C");
   gROOT->ProcessLine(".class A<C>");
   gROOT->ProcessLine(".class B");
   gROOT->ProcessLine(".class N::B");
   gROOT->ProcessLine(".class C");
   gROOT->ProcessLine(".class D");
   gROOT->ProcessLine(".U ConversionOp.C");
   
   gROOT->ProcessLine(".L ConversionOp.h+");
   gROOT->ProcessLine(".x ConversionOp.C");
   gROOT->ProcessLine(".class A<C>");
   gROOT->ProcessLine(".class B");
   gROOT->ProcessLine(".class N::B");
   gROOT->ProcessLine(".class C");
   gROOT->ProcessLine(".class D");
   
}

