/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <cstdio>
using namespace std;

namespace FOO{

class bug{
  private:
    int i;
    
  public:

    bug(int j) : i(j) { }
    int get() { return 0; }
    void set(int j) { i=j; }
    friend bug operator+(const bug& rhs, int c);
    friend bug operator>(const bug& rhs, int c);
  };
bug operator+(const bug& rhs, int c) { return(rhs); }
bug operator>(const bug& rhs, int c) { return(rhs); }
};


//Using the following LinkDef.

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace FOO;
#pragma link C++ nestedclasses;

#pragma link C++ class FOO::bug-!;
#pragma link C++ function FOO::operator+(const bug&, int); 
//#pragma link C++ function FOO::operator<(const bug&, int); 
#pragma link C++ function FOO::operator>(const bug&, int); 

#endif


