#if defined(ClingWorkAroundMissingSmartInclude) || defined(ClingWorkAroundMissingDynamicScope)
#include "TestClass.cc"
#else
#include "TestClass.cc+"
#endif
void readBack()
{
  TFile *outputB = TFile::Open("b.root");
  TFile *outputA = TFile::Open("a.root");
  TestClass *testB = (TestClass *) outputB->Get("testB");
  TestClass *testA = (TestClass *) outputA->Get("testA");
  if ( !testB ) {
    std::cout << "Couldn't find B?" << std::endl;
    return;
  }
  std::cout << "Readback for testB->GetRef()" << std::endl << (testB->GetRef() != nullptr) << std::endl;
  std::cout << "******************************************" << std::endl;
}

void test3()
{
  readBack();
}
