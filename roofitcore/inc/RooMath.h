/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   20-Jun-2000 DK Created initial version
 *   18-Jun-2001 WV Imported from RooFitTools
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/
#ifndef ROO_MATH
#define ROO_MATH

#include "RooFitCore/RooComplex.hh"

class RooMath {
public:
  static RooComplex ComplexErrFunc(Double_t re, Double_t im= 0);
  static RooComplex ComplexErrFunc(const RooComplex& z);
private:
  ClassDef(RooMath,0) // math utility routines
};

#endif
