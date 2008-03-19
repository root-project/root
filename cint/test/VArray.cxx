/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**********************************************************
* 
**********************************************************/

#include "VArray.h"

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
VArray::VArray()
{
  numElements = 0;
  adrAry = (VObject**)NULL;
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
VArray::VArray(VArray& obj)
{
  Int_t result;

  numElements = 0;
  adrAry = (VObject**)NULL;

  result = SetNumElements(obj.GetNumElements());
  if(SUCCESS!=result) return;

  for(Int_t i=0;i<numElements;i++) adrAry[i] = obj.adrAry[i];
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
VArray& VArray::operator=(VArray& obj)
{
  Int_t result;

  result = SetNumElements(obj.GetNumElements());
  if(SUCCESS!=result) return(*this);

  for(Int_t i=0;i<numElements;i++) adrAry[i] = obj.adrAry[i];

  return(*this);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
VArray::~VArray()
{
  if(adrAry) {
    delete[] adrAry;
    adrAry = (VObject**)NULL;
  }
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
Int_t VArray::SetNumElements(Int_t numin)
{
  VObject** storeAry = adrAry;
  Int_t storeNum = numElements;
  
  if(numin<0) return FAILURE;
  else if(0==numin) {
    if(adrAry) {
      delete[] adrAry;
      adrAry = (VObject**)NULL;
    }
    numElements = numin;
    return(SUCCESS);
  }


  numElements = numin;
  adrAry = new VObject*[numElements];
  if((VObject**)NULL==adrAry) {
    adrAry = storeAry;
    numElements = storeNum;
    return(FAILURE);
  }

  if(storeAry) {
    Int_t minNum = MIN(numElements,storeNum);
    Int_t i;
    for(i=0;i<minNum;i++) adrAry[i] = storeAry[i];
    delete[] storeAry;
  }

  return(SUCCESS);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
Int_t VArray::Add(VObject* obj,Int_t index)
{
  ///* DEBUG */ printf("Add(%p)\n",obj);
  VObject** storeAry = adrAry;
  void *ppp = adrAry;

  adrAry = new VObject*[++numElements];
  if((VObject**)NULL==adrAry) {
    adrAry = storeAry;
    --numElements;
    return(FAILURE);
  }

  if(index<0||numElements-1<index) index = numElements-1;

  if(storeAry) {
    Int_t i;
    for(i=0;i<index;i++) adrAry[i] = storeAry[i];
    ///*DEBUG*/ for(i=0;i<index;i++) printf("%d:%lx:%lx ",i,adrAry[i],storeAry[i]);
    ///*DEBUG*/printf("\n");
    adrAry[index] = obj;
    for(i=index+1;i<numElements;i++) adrAry[i] = storeAry[i-1];
    delete[] storeAry;
  }
  else {
    assert(0==index);
    adrAry[index] = obj;
  }

  ///* DEBUG */ printf("%lx.%d Add(%lx) = %lx\n",this,index,adrAry[index],obj);
  //G__pause();
  return(SUCCESS);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
VObject* VArray::Delete(Int_t index,Int_t flag)
{
  if(index<0||numElements-1<index) index=numElements-1;

  VObject** storeAry = adrAry;
  assert((VObject**)NULL!=adrAry);

  VObject *p = storeAry[index];

  if(flag) {
    delete p;
    p = (VObject*)NULL;
  }

  if(1==numElements) {
    numElements = 0;
    delete[] adrAry;
    adrAry = (VObject**)NULL;
    return(p);
  }

  assert(2<=numElements);

  adrAry = new VObject*[--numElements];
  if((VObject**)NULL==adrAry) {
    adrAry = storeAry;
    ++numElements;
    return(NULL);
  }

  if(storeAry) {
    Int_t i;
    for(i=0;i<index;i++) adrAry[i] = storeAry[i];
    for(i=index+1;i<numElements+1;i++) adrAry[i-1] = storeAry[i];
    delete[] storeAry;
  }

  return(p);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
VObject* VArray::GetElement(Int_t index)
{
  if(index<0 || numElements<=index) {
    assert(0);
    return(adrAry[0]);
  }

  ///* DEBUG */ printf("%lx.%d GetElement(%lx)\n",this,index,adrAry[index]);
  //G__pause();
  return(adrAry[index]);
}

