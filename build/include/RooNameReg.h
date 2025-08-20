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

// String name registry
class RooNameReg : public TNamed {
public:

  RooNameReg(const RooNameReg& other) = delete;

  static RooNameReg& instance() ;
  const TNamed* constPtr(const char* stringPtr) ;
  /// Return C++ string corresponding to given TNamed pointer.
  inline static const char* constStr(const TNamed* ptr) {
    return ptr ? ptr->GetName() : nullptr;
  }
  static const TNamed* ptr(const char* stringPtr) ;
  /// Return C++ string corresponding to given TNamed pointer.
  inline static const char* str(const TNamed* ptr) {
    return ptr ? ptr->GetName() : nullptr;
  }
  static const TNamed* known(const char* stringPtr) ;
  static const std::size_t& renameCounter() ;

  enum {
    kRenamedArg = BIT(19)    ///< TNamed flag to indicate that some RooAbsArg has been renamed (flag set in new name)
  };

protected:
  RooNameReg();

  friend class RooAbsArg;
  friend class RooAbsData;
  static void incrementRenameCounter() ;

  std::unordered_map<std::string,std::unique_ptr<TNamed>> _map;
  std::size_t _renameCounter = 0;
};

#endif


