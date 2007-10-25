void runConversionOp() {
   gSystem->Setenv("LINES","-1");

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

