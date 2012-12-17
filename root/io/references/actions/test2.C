#if defined(ClingWorkAroundMissingSmartInclude) || defined(ClingWorkAroundMissingDynamicScope)
#include "TestClass.cc"
#else
#include "TestClass.cc+"
#endif
void readBack() 
{
  TFile* outputB = TFile::Open("b.root");
  TestClass* testB = (TestClass*) outputB->Get("testB");
  if ( !testB ) { cout << "Couldn't find B?" << endl; return; }

#ifdef ClingWorkAroundJITandInline
   cout.setf(ios_base::hex, ios_base::basefield);
   cout << "fBits settings B: 0x" << testB->TestRefBits(0xFFFFFFFF) << endl;
#else
   cout << "fBits settings B: 0x" << hex << testB->TestRefBits(0xFFFFFFFF) << endl;
#endif
   
  cout << "Readback for testB->GetRef()" << std::endl << testB->GetRef() << endl;
  cout << "******************************************" << std::endl;
}

void test2() 
{
 
   readBack(); 

}
