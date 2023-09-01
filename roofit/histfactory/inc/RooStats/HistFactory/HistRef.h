// @(#)root/roostats:$Id$
// Author: L. Moneta
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef HISTFACTORY_HISTREF_H
#define HISTFACTORY_HISTREF_H

#include <memory>

class TH1;

namespace RooStats{
namespace HistFactory {


// Internal class wrapping an histogram and managing its content.
// conveninet for dealing with histogram pointers in the
// HistFactory class
class HistRef {

public:


   /// constructor - use gives away ownerhip of the given pointer
   HistRef(TH1 * h = nullptr) : fHist(h) {}

   HistRef( const HistRef& other ) :
   fHist() {
      if (other.fHist) fHist.reset(CopyObject(other.fHist.get()));
   }

   HistRef(HistRef&& other) :
   fHist(std::move(other.fHist)) {}

   ~HistRef() {}

   /// assignment operator (delete previous contained histogram)
   HistRef & operator= (const HistRef & other) {
      if (this == &other) return *this;

      fHist.reset(CopyObject(other.fHist.get()));
      return *this;
   }

   HistRef& operator=(HistRef&& other) {
     fHist = std::move(other.fHist);
     return *this;
   }

   TH1 * GetObject() const { return fHist.get(); }

   /// set the object - user gives away the ownerhisp
   void SetObject(TH1 *h)  {
      fHist.reset(h);
   }

   /// operator= passing an object pointer :  user gives away its ownerhisp
   void operator= (TH1 * h) { SetObject(h); }

   /// Release ownership of object.
   TH1* ReleaseObject() {
     return fHist.release();
   }



private:
   static TH1 * CopyObject(const TH1 * h);
   std::unique_ptr<TH1> fHist;   ///< pointer to contained histogram
};

}
}

#endif
