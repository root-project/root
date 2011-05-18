/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMapCatEntry.h,v 1.13 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_MAP_CAT_ENTRY
#define ROO_MAP_CAT_ENTRY

#include "Riosfwd.h"
#include "TNamed.h"
#include "TRegexp.h"
#include "RooCatType.h"

class RooMapCatEntry : public TNamed {
public:
  inline RooMapCatEntry() : TNamed(), _regexp(""), _cat() {} 
  virtual ~RooMapCatEntry() {} ;
  RooMapCatEntry(const char* exp, const RooCatType* cat) ;
  RooMapCatEntry(const RooMapCatEntry& other) ;
  virtual TObject* Clone(const char* newName=0) const { 
 	RooMapCatEntry* ca = new RooMapCatEntry(*this); 
	if (newName) { ca->SetName(newName) ; }
	return ca ;
	}

  inline Bool_t ok() { return (_regexp.Status()==TRegexp::kOK) ; }
  Bool_t match(const char* testPattern) const ;
  inline const RooCatType& outCat() const { return _cat ; }

protected:

  TString mangle(const char* exp) const ;

  TRegexp _regexp ;
  RooCatType _cat ;
	
  ClassDef(RooMapCatEntry,1) // Utility class, holding a map expression from a index label regexp to a RooCatType
} ;


#endif
