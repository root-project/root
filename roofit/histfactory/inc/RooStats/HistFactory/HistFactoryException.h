// @(#)root/roostats:$Id$
// Author: George Lewis, Kyle Cranmer
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef HISTFACTORY_EXCEPTION
#define HISTFACTORY_EXCEPTION

#include <iostream>
#include <exception>


namespace RooStats{
  namespace HistFactory{

    class hf_exc: public std::exception
    {
      virtual const char* what() const throw()
      {
	return "HistFactory - Exception";
      }
    };

  }
}

//static hf_exc bad_hf;

#endif
