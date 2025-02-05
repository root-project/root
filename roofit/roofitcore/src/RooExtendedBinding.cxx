/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <iostream>
#include <cmath>

#include "RooExtendedBinding.h"
#include "RooAbsPdf.h"
#include "RooAbsCategory.h"


 RooExtendedBinding::RooExtendedBinding(const char *name, const char *title, RooAbsPdf& _pdf) :
   RooAbsReal(name,title),
   pdf("pdf","pdf",this,_pdf)
 {
 }


 RooExtendedBinding::RooExtendedBinding(const RooExtendedBinding& other, const char* name) :
   RooAbsReal(other,name),
   pdf("pdf",this,other.pdf)
 {
 }



 double RooExtendedBinding::evaluate() const
 {
   // ENTER EXPRESSION IN TERMS OF VARIABLE ARGUMENTS HERE
   return (const_cast<RooAbsPdf &>(static_cast<RooAbsPdf const&>(pdf.arg()))).expectedEvents(nullptr) ;
 }



