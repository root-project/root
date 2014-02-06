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


class TH1; 

namespace RooStats{
namespace HistFactory {


// Internal class wrapping an histogram and managing its content. 
// conveninet for dealing with histogram pointers in the 
// HistFactory class 
class HistRef {
  
public:


   // constructor - use gives away ownerhip of the given pointer
   HistRef(TH1 * h = 0) : fHist(h) {}

   HistRef( const HistRef& other ) : 
   fHist(0) { 
      if (other.fHist) fHist = CopyObject(other.fHist); 
   }

   ~HistRef() { 
      DeleteObject(fHist); 
   }
 
   // assignment operator (delete previous contained histogram)
   HistRef & operator= (const HistRef & other) { 
      if (this == &other) return *this; 
      DeleteObject(fHist); 
      fHist = CopyObject(other.fHist);
      return *this;
   }

   TH1 * GetObject() const { return fHist; }

   // set the object - user gives away the ownerhisp 
   void SetObject(TH1 *h)  { 
      DeleteObject(fHist);
      fHist = h;
   }

   // operator= passing an object pointer :  user gives away its ownerhisp 
   void operator= (TH1 * h) { SetObject(h); } 

   static TH1 * CopyObject(TH1 * h); 
   static void  DeleteObject(TH1 * h); 
   
protected: 
   
   TH1 * fHist;   // pointer to contained histogram 
};

}
}

#endif
