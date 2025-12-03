#include <vector>
#include <string>
#include <iostream>
#include "TSelector.h"
#include "TError.h"

#ifndef __CINT__
#include "emptysel.h"
#endif

void execGetSelector(string infilename = "Event1.root", int nevents = 10)
{
  infilename.append("/T1");
  
  TSelector *sel = TSelector::GetSelector("emptysel.C+O");
  //delete sel;
  //dummy * mydummy = (dummy*)TSelector::GetSelector("dummy.C+");;
  emptysel * mydummy = (emptysel*)sel;

  cout << "In Macro after GetSelector" << endl;
  mydummy->m_testvar = 10; 
  mydummy->printAddress();
  cout << "Testvar before Process: " << mydummy->m_testvar << endl << endl;
 
  //myAnalysis->setGoodRunList("goodruns.xml");
  
  TChain f;
  f.Add(infilename.c_str());
  gErrorIgnoreLevel = kError;
  f.Process(mydummy,"",nevents,2);
  
}
