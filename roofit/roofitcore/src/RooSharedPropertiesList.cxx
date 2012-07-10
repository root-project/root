/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Class RooSharedPropertiesList maintains the properties of RooRealVars
// and RooCategories that are clones of each other.
// END_HTML
//

#include "RooFit.h"
#include "RooSharedPropertiesList.h"
#include "RooSharedProperties.h"
#include "RooLinkedListIter.h"
#include "TIterator.h"
#include "RooMsgService.h"
#include "Riostream.h"
using std::cout ;
using std::endl ;

using namespace std;

ClassImp(RooSharedPropertiesList)
;



//_____________________________________________________________________________
RooSharedPropertiesList::RooSharedPropertiesList() 
{
  // Constructor
} 



//_____________________________________________________________________________
RooSharedPropertiesList::~RooSharedPropertiesList() 
{
  // Destructor

  // Delete all objects in property list
  RooFIter iter = _propList.fwdIterator() ;
  RooSharedProperties* prop ;
  while((prop=(RooSharedProperties*)iter.next())) {
    delete prop ;
  }
} 



//_____________________________________________________________________________
RooSharedProperties* RooSharedPropertiesList::registerProperties(RooSharedProperties* prop, Bool_t canDeleteIncoming) 
{
  // Register property into list and take ownership. 
  //
  // If an existing entry has a UUID that identical to that of the argument prop, 
  // the argument prop is deleted and a pointer to the already stored is returned to
  // eliminate the duplication of instances with a unique identity.
  //
  // The caller should therefore not refer anymore to the input argument pointer as
  // as the object cannot be assumed to be live.

  if (prop==0) {
    oocoutE((TObject*)0,InputArguments) << "RooSharedPropertiesList::ERROR null pointer!:" << endl ;
    return 0 ;
  }


  // If the reference count is non-zero, it is already in the list, so no need
  // to look it up anymore
  if (prop->inSharedList()) {
    prop->increaseRefCount() ;
    return prop ;
  }

  // Find property with identical uuid in list
  RooFIter iter = _propList.fwdIterator() ;
  RooSharedProperties* tmp ;
  while((tmp=(RooSharedProperties*)iter.next())) {
    if (tmp != prop && *tmp==*prop) {
      // Found another instance of object with identical UUID 

      // Delete incoming instance, increase ref count of already stored instance
      // cout << "RooSharedProperties::reg deleting incoming prop " << prop << " recycling existing prop " << tmp << endl ;

      // Check if prop is in _propList
      if (_propList.FindObject(prop)) {
	// cout << "incoming object to be deleted is in proplist!!" << endl ;
      } else {
	// cout << "deleting prop object " << prop << endl ;
	if (canDeleteIncoming) delete prop ;
      }

      // delete prop ;
      //_propList.Add(tmp) ;
      tmp->increaseRefCount() ;

      // Return pointer to already-stored instance
      return tmp ;
    }
  }

  
  // cout << "RooSharedProperties::reg storing incoming prop " << prop << endl ;
  prop->setInSharedList() ;
  prop->increaseRefCount() ;
  _propList.Add(prop) ;
  return prop ;
}



//_____________________________________________________________________________
void RooSharedPropertiesList::unregisterProperties(RooSharedProperties* prop) 
{
  // Decrease reference count of property. If reference count is at zero,
  // delete the propery

  prop->decreaseRefCount() ;

  if (prop->refCount()==0) {
    _propList.Remove(prop) ;
    
    // We own object if ref-counted list. If count drops to zero, delete object
    delete prop ;
  }
  
}


