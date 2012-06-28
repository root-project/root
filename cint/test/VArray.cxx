/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "VArray.h"

#include <algorithm>
#include <cassert>

using namespace std;

VArray::VArray()
: numElements(0)
, adrAry(0)
{
}

VArray::VArray(const VArray& obj)
: numElements(0)
, adrAry(0)
{
   int result = SetNumElements(obj.GetNumElements());
   if (result != SUCCESS) {
      return;
   }
   for (int i = 0; i < numElements; ++i) {
      adrAry[i] = obj.adrAry[i];
   }
}

VArray& VArray::operator=(const VArray& obj)
{
   if (this != &obj) {
      int result = SetNumElements(obj.GetNumElements());
      if (result != SUCCESS) {
         return *this;
      }
      for (int i = 0; i < numElements; ++i) {
         adrAry[i] = obj.adrAry[i];
      }
   }
   return *this;
}

VArray::~VArray()
{
   delete[] adrAry;
   adrAry = 0;
}

int VArray::SetNumElements(int numin)
{
   VObject** storeAry = adrAry;
   int storeNum = numElements;
   if (numin < 0) {
      return FAILURE;
   }
   else if (!numin) {
      delete[] adrAry;
      adrAry = 0;
      numElements = numin;
      return SUCCESS;
   }
   numElements = numin;
   adrAry = new VObject*[numElements];
   if (!adrAry) {
      adrAry = storeAry;
      numElements = storeNum;
      return FAILURE;
   }
   if (storeAry) {
      int minNum = min(numElements, storeNum);
      for (int i = 0; i < minNum; ++i) {
         adrAry[i] = storeAry[i];
      }
      delete[] storeAry;
   }
   return SUCCESS;
}

int VArray::Add(VObject* obj, int index)
{
   VObject** storeAry = adrAry;
   void *ppp = adrAry;
   adrAry = new VObject*[++numElements];
   if (!adrAry) {
      adrAry = storeAry;
      --numElements;
      return FAILURE;
   }
   if ((index < 0) || ((numElements - 1) < index)) {
      index = numElements - 1;
   }
   if (storeAry) {
      for (int i = 0; i < index; ++i) {
         adrAry[i] = storeAry[i];
      }
      adrAry[index] = obj;
      for (int i = index + 1; i < numElements; ++i) {
         adrAry[i] = storeAry[i-1];
      }
      delete[] storeAry;
   }
   else {
      assert(index == 0);
      adrAry[index] = obj;
   }
   return SUCCESS;
}

VObject* VArray::Delete(int index, int flag)
{
   if ((index < 0) || ((numElements - 1) < index)) {
      index = numElements - 1;
   }
   VObject** storeAry = adrAry;
   assert(adrAry != 0);
   VObject* p = storeAry[index];
   if (flag) {
      delete p;
      p = 0;
   }
   if (numElements == 1) {
      numElements = 0;
      delete[] adrAry;
      adrAry = 0;
      return p;
   }
   assert(numElements >= 2);
   adrAry = new VObject*[--numElements];
   if (!adrAry) {
      adrAry = storeAry;
      ++numElements;
      return 0;
   }
   if (storeAry) {
      for (int i = 0; i < index; ++i) {
         adrAry[i] = storeAry[i];
      }
      for (int i = index + 1; i < (numElements + 1); ++i) {
         adrAry[i-1] = storeAry[i];
      }
      delete[] storeAry;
   }
   return p;
}

VObject* VArray::GetElement(int index)
{
   if ((index < 0) || (numElements <= index)) {
      assert(0);
      return adrAry[0];
   }
   return adrAry[index];
}

