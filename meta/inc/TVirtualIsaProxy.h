// @(#)root/meta:$Name:  $:$Id: TClass.h,v 1.49 2005/03/20 21:25:12 brun Exp $
// Author: Markus Frank 20/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualIsaProxy
#define ROOT_TVirtualIsaProxy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClass                                                               //
//                                                                      //
// Virtual IsaProxy base class.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TVirtualIsaProxy  {
public:
   virtual void SetClass(TClass *cl) = 0;
   virtual TClass* operator()(const void *obj) = 0;
};

#endif // ROOT_TVirtualIsaProxy
