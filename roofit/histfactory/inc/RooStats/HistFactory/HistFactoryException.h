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

#include <exception>
#include <string>


namespace RooStats{
  namespace HistFactory{

    class hf_exc: public std::exception
    {
    public:
      hf_exc(std::string message = "") : _message("HistFactory - Exception " + message) { }

      virtual const char* what() const noexcept
      {
        return _message.c_str();
      }

    private:
      const std::string _message;
    };

  }
}

#endif
