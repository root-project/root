//____________________________________________________________________
// 
// 
// 

//____________________________________________________________________
//
// $Id$
// $Author: pcanal $
// $Date: 2002/08/02 22:24:36 $
// $Copyright: (C) 2002 BRAHMS Collaboration <brahmlib@rhic.bnl.gov>
//
#ifndef TREEPROBLEM_Foo
#include "Foo.h"
#endif
#ifndef __IOSTREAM__
#include <iostream>
#endif

using namespace std;
//____________________________________________________________________
ClassImp(Foo);

//____________________________________________________________________
void Foo::Print(Option_t* option) const 
{
  cout << "Foo class: " << fFoo << endl;
}

//____________________________________________________________________
//
// $Log: Foo.cxx,v $
// Revision 1.1.1.1  2002/08/02 22:24:36  pcanal
// Initial import of my suite of test for root and cint
//
//
