#ifndef TestClass_cc
#define TestClass_cc

#include "TestClass.hh"
#include "TObjArray.h"
#include "TExec.h"
ClassImp(TestClass)

TestClass::TestClass(const std::string& name) : TObject(), fName(name)
{

  // Adding this to the list of execs
  TRef::AddExec("TestClass::CallOnDemand()");
  fReference.SetAction( "TestClass::CallOnDemand()" );
}

void TestClass::CallOnDemand() 
{
  std::cout << "Test class called on demand" << std::endl;
}

#endif