#if defined(ClingWorkAroundMissingSmartInclude) || defined(ClingWorkAroundMissingDynamicScope)
#include "TestClass.cc"
#else
#include "TestClass.cc+"
#endif

void build() 
{
  TFile* outputA = TFile::Open("a.root", "recreate");
  TestClass* testA = new TestClass("A");
  TestClass* testB = new TestClass("B");

  TFile* outputB = TFile::Open("b.root", "recreate");

  testA->SetRef( testB );
  testB->SetRef( testA );

  outputA->cd();
  testA->Write("testA");

  outputB->cd();
  testB->Write("testB");

  outputA->Close();
  outputB->Close();
#ifdef ClingWorkAroundJITandInline
  cout.setf(ios_base::hex, ios_base::basefield);
  cout << "fBits settings A: 0x" << testA->TestRefBits(0xFFFFFFFF) << endl;
  cout << "fBits settings B: 0x" << testB->TestRefBits(0xFFFFFFFF) << endl;
#else
  cout << "fBits settings A: 0x" << hex << testA->TestRefBits(0xFFFFFFFF) << endl;
  cout << "fBits settings B: 0x" << hex << testB->TestRefBits(0xFFFFFFFF) << endl;
#endif
  delete outputA;
  delete outputB;
}


void test() 
{
 
  build(); 

}
