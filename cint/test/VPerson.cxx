/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif
#include "VPerson.h"

VPerson::VPerson() : VObject() , name() , syozoku()
{
}

VPerson::VPerson(VPerson& x) : VObject() , name(x.name) , syozoku(x.syozoku)
{
}

VPerson::VPerson(Char_t* nameIn,Char_t* syozokuIn) : VObject() , name(nameIn) , syozoku(syozokuIn)
{
}

VPerson::VPerson(Char_t* nameIn,Int_t num) : VObject(), name(nameIn), syozoku()
{
  char buf[10];
  sprintf(buf,"syozoku%d",num);
  syozoku = buf;
}

VPerson& VPerson::operator=(VPerson& x) 
{
  name = x.name;
  syozoku = x.syozoku;
  return x;
}

VPerson::~VPerson() 
{
}

void VPerson::set(Char_t* nameIn,Char_t* syozokuIn)
{
  name = nameIn;
  syozoku = syozokuIn;
}

void VPerson::disp() {
  cout << name.String() << " " << syozoku.String() << endl;  
}
