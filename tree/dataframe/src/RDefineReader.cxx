// Author: Enrico Guiraud, CERN  08/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDF/RDefineReader.hxx>
#include <ROOT/RDF/RDefineBase.hxx>
#include <ROOT/RDF/Utils.hxx> // TypeID2TypeName
#include <TClass.h>

#include <stdexcept> // std::runtime_error
#include <string>
#include <typeinfo>

