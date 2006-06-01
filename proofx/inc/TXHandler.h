// @(#)root/proofx:$Name:  $:$Id: TXHandler.h,v 1.1 2006/04/19 10:50:01 rdm Exp $
// Author: G. Ganis Mar 2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXHandler
#define ROOT_TXHandler

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXHandler                                                            //
//                                                                      //
// Handler of asynchronous events for xproofd sockets.                  //
// Classes which need this should inherit from it and overload the      //
// relevant methods.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TXHandler {

public:
   TXHandler() { }
   virtual ~TXHandler() { }

   virtual Bool_t HandleInput();
   virtual Bool_t HandleError();

   ClassDef(TXHandler, 0) //Template class for handling of async events
};

#endif
