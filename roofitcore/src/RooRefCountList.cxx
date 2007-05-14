/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooRefCountList.cxx,v 1.10 2007/05/11 09:11:58 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// A RooRefCountList is a RooLinkedList that keeps a reference counter
// with each added node. Multiple Add()s of the same object will increase
// the counter instead of adding multiple copies. Remove() decrements the 
// reference count until zero, when the object is actually removed.

#include "RooFit.h"

#include "RooRefCountList.h"
#include "RooRefCountList.h"

#include "Riostream.h"
#include <stdlib.h>

ClassImp(RooRefCountList)
  ;


RooRefCountList::RooRefCountList()
  : RooLinkedList(17) 
{ 
}


void RooRefCountList::Add(TObject* obj, Int_t count) 
{
  // Check if we already have it
  TObject* listObj = FindObject(obj) ;
  if (!listObj) {
    // Add to list with reference count 
    RooLinkedList::Add(obj, count) ;
    //cout << "RooRefCountList::AddLast(" << obj << ") adding object" << endl ;
  } else {
    RooLinkedListElem* link = findLink(obj) ;
    while(count--) link->incRefCount() ;    
    //cout << "RooRefCountList::AddLast(" << obj << ") incremented reference count to " << link->refCount() << endl ;
  }

}


Bool_t RooRefCountList::Remove(TObject* obj) 
{
  RooLinkedListElem* link = findLink(obj) ;
  if (!link) {
    return 0 ;
  } else {
    if (link->decRefCount()==0) {
      //cout << "RooRefCountList::AddLast(" << obj << ") removed object" << endl ;
      return RooLinkedList::Remove(obj) ;
    }
    //cout << "RooRefCountList::AddLast(" << obj << ") decremented reference count to " << link->refCount() << endl ;
  }
  return 0 ;
}


Bool_t RooRefCountList::RemoveAll(TObject* obj)
{
  return RooLinkedList::Remove(obj) ;
}


Int_t RooRefCountList::refCount(TObject* obj) 
{
  RooLinkedListElem* link = findLink(obj) ;
  if (!link) {
    return 0 ;
  } else {
    return link->refCount() ;
  }
}
