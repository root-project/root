/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***********************************************************************
* VArray.h , C++
*
************************************************************************
* Description:
*
***********************************************************************/

#ifndef VARRAY_H
#define VARRAY_H

#include "VType.h"
#include "VObject.h"

#define INVALIDINDEX -1

/**********************************************************
* Polymorphic Array object
**********************************************************/

class VArray : public VObject {
 public:

  VArray();
  VArray(VArray& obj);
  VArray& operator=(VArray& obj); 
  ~VArray();
  // Int_t operator==(VArray& x);

  Int_t SetNumElements(Int_t numin);
  Int_t Add(VObject* obj,Int_t index=INVALIDINDEX);
  VObject* Delete(Int_t index=INVALIDINDEX,Int_t flag=0);

  Int_t GetNumElements() { return numElements;}
  VObject* GetElement(Int_t index);
  VObject& operator[](Int_t index) { return(*GetElement(index)); }

 private:
  Int_t numElements;  
  VObject**    adrAry;
};

#endif

