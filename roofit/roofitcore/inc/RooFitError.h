/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFitCore_RooFitError_h
#define RooFitCore_RooFitError_h

#include <ostream>
#include <sstream>
#include <stdexcept>

class RooFitError {

public:
   RooFitError(std::ostream& os) : _os{os} {}

   operator std::ostream &() { return _errMsgStream; }

   inline void logAndThrow() const {
      const std::string errMsg = _errMsgStream.str();
      _os << errMsg << std::endl;
      throw std::runtime_error(errMsg);
   }

private:
   std::ostream& _os;
   std::stringstream _errMsgStream;
};

#endif
