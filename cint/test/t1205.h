/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

// compiled.h ,  precompiled class header file
#ifndef COMPILED_H
#define COMPILED_H

#include <stdio.h>
#include <vector>

using namespace std; 

typedef vector < double * > DList;
typedef vector < DList * > :: iterator DListIterator;
//typedef vector<double*> DList;
//typedef vector<DList*>::iterator DListIterator;
// typedef double DListIterator;

// ----------------------------------------------------------

class Compiled1 {
 public:
  Compiled1();
  virtual void publicFunc1( DListIterator &iter );
  long G__virtualinfo;
};

// ----------------------------------------------------------

class Helper {
 public:
  Helper();

 void Execute( Compiled1 *ptr );
};

// compiled.cxx , precompiled class/function definition

//#include "compiled.h"

///////////////////////////////////////////////////////////////////////
// Compiled1 member function
///////////////////////////////////////////////////////////////////////
Compiled1::Compiled1() {
	return;
}

void Compiled1::publicFunc1( DListIterator &iter ) { 
  printf("Called Compiled1::publicFunc1()\n");
}


///////////////////////////////////////////////////////////////////////
// Helper member functions
///////////////////////////////////////////////////////////////////////
Helper::Helper() {
  return;
}

void Helper::Execute( Compiled1 *ptr )
{
  DListIterator iter;
  ptr->publicFunc1(iter); 
}


#endif

#ifndef STUB_H
#define STUB_H

//#include "compiled.h"

/////////////////////////////////////////////////////////////////////////
// Interface of Stub1 class is compiled so that 
//  - We can use this class in a compiled code
//  - We can resolve virtual function 
// However, body of member functions are interpreted.
/////////////////////////////////////////////////////////////////////////

class Stub1 : public Compiled1 { 
 public:
  Stub1() {}
  virtual void publicFunc1( DListIterator &iter ); 

  long G__virtualinfo; // new feature, this allows you to inherit an
                       // interpreted class from a Stub class

};

#endif

#ifdef __MAKECINT__
#pragma stub C++ function Stub1::publicFunc1;
#endif

