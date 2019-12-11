/// \file RError.cxx
/// \ingroup Base ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-12-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RError.hxx"

thread_local bool ROOT::Experimental::Detail::RStatusBase::fgThrowInstantExceptions = true;

void ROOT::Experimental::SetThrowInstantExceptions(bool value)
{
   Detail::RStatusBase::SetThrowInstantExceptions(value);
}
