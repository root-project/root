/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// A RooRefCountList is a THashList that keeps a reference counter
// with each added node. Multiple Add()s of the same object will increase
// the counter instead of adding multiple copies. Remove() decrements the 
// reference count until zero, when the object is actually removed.

#include "RooFitCore/RooRefCountList.hh"

#include <iostream.h>
#include <stdlib.h>

ClassImp(RooRefCountList)
  ;


void RooRefCountList::AddLast(TObject* obj) 
{
  // Check if we already have it
  Int_t idx ;
  TObjLink* link = FindLink(obj,idx) ;
  if (!link) {
    // Add to list with reference count 1
    THashList::AddLast(obj,"1") ;
    //cout << "RooRefCountList::AddLast(" << obj << ") adding object" << endl ;
  } else {
    const char* opt = link->GetOption() ;
    Int_t count = atoi(opt) ;
    link->SetOption(Form("%d",++count)) ;
    //cout << "RooRefCountList::AddLast(" << obj << ") incremented reference count to " << count << endl ;
  }

}


TObject* RooRefCountList::Remove(TObject* obj) 
{
  Int_t idx ;
  TObjLink* link = FindLink(obj,idx) ;
  if (!link) {
    return 0 ;
  } else {
    const char* opt = link->GetOption() ;
    Int_t count = atoi(opt) ;
    if (count==1) {
      //cout << "RooRefCountList::AddLast(" << obj << ") removed object" << endl ;
      return THashList::Remove(obj) ;
    }

    link->SetOption(Form("%d",--count)) ;
    //cout << "RooRefCountList::AddLast(" << obj << ") decremented reference count to " << count << endl ;
  }
  return obj ;
}


Int_t RooRefCountList::refCount(TObject* obj) 
{
  Int_t idx ;
   TObjLink* link = FindLink(obj,idx) ;
   if (!link) {
     return 0 ;
   } else {
     const char* opt = link->GetOption() ;
     Int_t count = atoi(opt) ;
     return count ;
   }
}


TObject* RooRefCountList::RemoveAll(TObject* obj) 
{
  return THashList::Remove(obj) ;
}
