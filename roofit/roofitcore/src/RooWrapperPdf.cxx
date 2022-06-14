// Author: Stephan Hageboeck, CERN  12 Sep 2019

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2019, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
 * \class RooWrapperPdf
 * The RooWrapperPdf is a class that can be used to convert a function into a PDF.
 *
 * During construction, a RooAbsReal has to be passed. When this function is evaluated, the wrapper pdf will
 * in addition evaluate its integral, and normalise the returned value. It will further ensure that negative
 * return values are clipped at zero.
 *
 * Functions calls such as analytical integral requests or plot sampling hints are simply forwarded to the RooAbsReal
 * that was passed in the constructor.
 */


#include "RooWrapperPdf.h"

ClassImp(RooWrapperPdf)


