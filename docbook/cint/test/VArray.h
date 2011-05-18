/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef VARRAY_H
#define VARRAY_H

#include "VObject.h"
#include "VType.h"

#define INVALIDINDEX -1

class VArray : public VObject
{
   // Polymorphic Array object
public:

   VArray();
   VArray(const VArray& obj);
   VArray& operator=(const VArray& obj);
   ~VArray();

   int SetNumElements(int numin);
   int Add(VObject* obj, int index = INVALIDINDEX);
   VObject* Delete(int index = INVALIDINDEX, int flag = 0);

   int GetNumElements() const
   {
      return numElements;
   }

   VObject* GetElement(int index);

   VObject& operator[](int index)
   {
      return *GetElement(index);
   }

private:
   int numElements;
   VObject** adrAry;
};

#endif // VARRAY_H
