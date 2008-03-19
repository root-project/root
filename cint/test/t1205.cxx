/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1205.h"
#endif

// -----------------------------------------------------------------------

// lets create a derived stub
class DerivedStub : public Stub1  { 
 public:
  DerivedStub() {};
  void publicFunc1( DListIterator &iter ); 
};


void DerivedStub::publicFunc1(  DListIterator &iter ) {
  printf("Called DerivedStub::publicFunc1()\n");
}

// -----------------------------------------------------------------------

// Body of Stub1 member function
// giving definition to Stub1 virtual function
void Stub1::publicFunc1( DListIterator &iter ) {
  printf("Called Stub1::publicFunc1()\n");
}

// -----------------------------------------------------------------------


/////////////////////////////////////////////////////////////////////////
int main() 
{
  Compiled1 *pC1 = new Compiled1(); 
//  Stub1 *pS = new Stub1(); 
  DerivedStub *pDS = new DerivedStub(); 

  Helper *pHelper = new Helper(); 

  printf( "Calling compiled func (will call Virtual publicFunc1)\n" );

  pHelper->Execute( pC1 );
//  pHelper->Execute( pS );
  pHelper->Execute( pDS );

  delete pC1;
//  delete pS;
  delete pDS;

  delete pHelper;

  return 0;
}

