// @(#)root/base:$Id$
// Author: Philippe Canal 09/2014

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQObjectEmitVA
#define ROOT_TQObjectEmitVA

// TQObject::EmitVA is implemented in its own header to break the
// circular dependency between TQObject and TQConnection.

#ifndef ROOT_TQObject
#include "TQObject.h"
#endif
#ifndef ROOT_TQConnection
#include "TQConnection.h"
#endif

template <typename... T> inline
void TQObject::EmitVA(const char *signal_name, Int_t /* nargs */, const T&... params)
{
   // Activate signal with variable argument list.
   // For internal use and for var arg EmitVA() in RQ_OBJECT.h.

   if (fSignalsBlocked || fgAllSignalsBlocked) return;

   TList classSigLists;
   CollectClassSignalLists(classSigLists, IsA());

   if (classSigLists.IsEmpty() && !fListOfSignals)
      return;

   TString signal = CompressName(signal_name);

   TQConnection *connection = 0;

   // execute class signals
   TList *sigList;
   TIter  nextSigList(&classSigLists);
   while ((sigList = (TList*) nextSigList()))
   {
      TIter nextcl((TList*) sigList->FindObject(signal));
      while ((connection = (TQConnection*)nextcl())) {
         gTQSender = GetSender();
         connection->ExecuteMethod(params...);
      }
   }
   if (!fListOfSignals)
      return;

   // execute object signals
   TIter next((TList*) fListOfSignals->FindObject(signal));
   while (fListOfSignals && (connection = (TQConnection*)next())) {
      gTQSender = GetSender();
      connection->ExecuteMethod(params...);
   }
}

#endif