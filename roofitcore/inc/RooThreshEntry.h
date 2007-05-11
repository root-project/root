/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooThreshEntry.rdl,v 1.11 2005/12/08 13:19:58 wverkerke Exp $
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
#ifndef ROO_THRESH_ENTRY
#define ROO_THRESH_ENTRY

#include "Riostream.h"
#include "TNamed.h"
#include "RooCatType.h"

class RooThreshEntry : public TObject {
public:
  inline RooThreshEntry() : TObject(), _thresh(0), _cat() {} 
  virtual ~RooThreshEntry() {} ;
  RooThreshEntry(Double_t thresh, const RooCatType& cat) ;
  RooThreshEntry(const RooThreshEntry& other) ;
  virtual TObject* Clone(const char*) const { return new RooThreshEntry(*this); }

  virtual Int_t Compare(const TObject *) const ;
  virtual Bool_t IsSortable() const { return kTRUE ; }

  inline Double_t thresh() const { return _thresh ; }
  inline const RooCatType& cat() const { return _cat ; }

protected:

  Double_t _thresh ;
  RooCatType _cat ;
	
  ClassDef(RooThreshEntry,1) // Utility class, holding a threshold/category state pair
} ;


#endif
