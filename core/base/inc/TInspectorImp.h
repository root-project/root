// @(#)root/base:$Id$
// Author: Fons Rademakers   07/05/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TInspectorImp
#define ROOT_TInspectorImp


////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TInspectorImp                                                              //
//                                                                            //
// ABC describing GUI independent object inspector (abstration mainly needed  //
// for Win32. On X11 systems it currently uses a standard TCanvas).           //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"

class TObject;


class TInspectorImp {

public:
   TInspectorImp() { }
   TInspectorImp(const TObject *, UInt_t, UInt_t) { }
   virtual ~TInspectorImp() { }

   virtual void Hide() { }
   virtual void Show() { }

   ClassDef(TInspectorImp,0)  //GUI independent inspector abc
};

#endif
