/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <string>
#include <iostream>
#ifndef __hpux
using namespace std;
#endif

int main() {
  int i=0;
  while (i<4 ) {
    if ( i%2 == 1 ) {
      string foo( "foo" );  // remove this and it doesn't crash.
      //printf("%s\n",foo.c_str());
      cout << foo << " " << i << endl;
    }
    ++i;
  } 
  return 0;
}

#if 0
void scope() {
  TFile *fA = new TFile("rolf.root");
  
  TIter next( fA->GetListOfKeys() );
  TKey * k;
  while ( k = (TKey*) next() ) {
    cout << "className="<<k->GetClassName()<<endl;
    if ( strcmp( k->GetClassName() , "TH1F" ) == 0 ) {
      TString foo( "foo" );  // remove this and it doesn't crash.
      //TString *foo = new TString("foo");  delete foo;// this is OK
    }
  } 
}

t749() {
  int i=0;
  while (i<10 ) {
    if ( i%2 ) {
      TString foo( "foo" );  // remove this and it doesn't crash.
      //TString *foo = new TString("foo");  delete foo;// this is OK
    }
    ++i;
  } 
}
#endif

