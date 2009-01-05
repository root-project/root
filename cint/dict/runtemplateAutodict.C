#include "TError.h"

int runtemplateAutodict()
{
  //gDebug = 7;
   gErrorIgnoreLevel = kWarning;

  gROOT->ProcessLine(".x simpleVectorTest.C");
  //gROOT->ProcessLine(".U simpleVectorTest.C");
  
  gROOT->ProcessLine(".x complexVectorTest.C");
  //gROOT->ProcessLine(".U complexVectorTest.C")
  
  gROOT->ProcessLine(".x complexVectorTest2.C");
  //gROOT->ProcessLine(".U complexVectorTest2.C");

  gROOT->ProcessLine(".x mapTest.C");
  //gROOT->ProcessLine(".U mapTest.C");

  gROOT->ProcessLine(".x templateClassTest.C");
  //gROOT->ProcessLine(".U templateClassTest.C");
  return 0;
}
