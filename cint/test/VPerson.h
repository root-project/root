/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

//#pragma includepath "\mpn\ke\proto3"
#include "VString.h"
#include "VObject.h"

class VPerson : public VObject {
 public:
  VPerson();
  VPerson(VPerson& x);
  VPerson(Char_t* nameIn,Char_t* syozokuIn);
  VPerson(Char_t* nameIn,Int_t num);
  VPerson& operator=(VPerson& x);
  ~VPerson();
  void set(Char_t* nameIn,Char_t* shozokuIn);
  Char_t* Name() { return(name.String()); }
  Char_t* Syozoku() { return(syozoku.String()); }
  void disp();
 private:
  VString name;
  VString syozoku;
};

