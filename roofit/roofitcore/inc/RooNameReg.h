/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNameReg.h,v 1.3 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_NAME_REG
#define ROO_NAME_REG

#include "TNamed.h"

#include <memory>
#include <unordered_map>
#include <string>

class RooNameReg : public TNamed {
public:

  static RooNameReg& instance() ;
  ~RooNameReg() override;
  const TNamed* constPtr(const char* stringPtr) ;
  const char* constStr(const TNamed* namePtr) ;
  static const TNamed* ptr(const char* stringPtr) ;
  static const char* str(const TNamed* ptr) ;
  static const TNamed* known(const char* stringPtr) ;
  static const std::size_t& renameCounter() ;

  enum {
    kRenamedArg = BIT(19)    ///< TNamed flag to indicate that some RooAbsArg has been renamed (flag set in new name)
  };

protected:
  RooNameReg();
//  RooNameReg(Int_t hashSize = 31) ;
  RooNameReg(const RooNameReg& other) = delete;

  friend class RooAbsArg;
  friend class RooAbsData;
  static void incrementRenameCounter() ;

  std::unordered_map<std::string,std::unique_ptr<TNamed>> _map;
  std::size_t _renameCounter = 0;

//  ClassDefOverride(RooNameReg,1) // String name registry
};

#endif


