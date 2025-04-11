//____________________________________________________________________
// 
// 
// 

//____________________________________________________________________
//
// $Id$
// $Author$
// $Date$
// $Copyright: (C) 2002 BRAHMS Collaboration <brahmlib@rhic.bnl.gov>
//
#ifndef TREEPROBLEM_Foo
#include "Foo.h"
#endif
#ifndef __IOSTREAM__
#include <iostream>
#endif

using std::cout;
using std::endl;

//____________________________________________________________________
ClassImp(Foo);

//____________________________________________________________________
void Foo::Print(Option_t* option) const 
{
  cout << "Foo class: " << fFoo << endl;
}

//____________________________________________________________________
//
// $Log$
//
