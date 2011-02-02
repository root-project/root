#include "TestClass.cc+"

void readBack() 
{
  TFile* outputB = TFile::Open("b.root");
  TestClass* testB = (TestClass*) outputB->Get("testB");
  if ( !testB ) { cout << "Couldn't find B?" << endl; return; }
  cout << "fBits settings B: 0x" << hex << testB->TestRefBits(0xFFFFFFFF) << endl;
  cout << "Readback for testB->GetRef()" << std::endl << testB->GetRef() << endl;
  cout << "******************************************" << std::endl;
}

void test2() 
{
 
   readBack(); 

}
