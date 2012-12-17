#if defined(ClingWorkAroundMissingSmartInclude) || defined(ClingWorkAroundMissingDynamicScope)
#include "TestClass.cc"
#else
#include "TestClass.cc+"
#endif
void readBack() 
{
  TFile* outputB = TFile::Open("b.root");
  TFile* outputA = TFile::Open("a.root");
  TestClass* testB = (TestClass*) outputB->Get("testB");
  TestClass* testA = (TestClass*) outputA->Get("testA");
  if ( !testB ) { cout << "Couldn't find B?" << endl; return; }
  cout << "Readback for testB->GetRef()" << std::endl << (testB->GetRef() != 0) << endl;
  cout << "******************************************" << std::endl;
}

void test3() 
{
 
   readBack(); 

}
