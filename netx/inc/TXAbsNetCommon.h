// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXAbsNetCommon
#define ROOT_TXAbsNetCommon

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXAbsNetCommon                                                       //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
// Called when an obj inheriting from this class gets redirected in     //
// sending a command to the server                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TXUnsolicitedMsg.h"

class TXAbsNetCommon: public TXAbsUnsolicitedMsgHandler {
public:

  virtual Bool_t OpenFileWhenRedirected(char *newfhandle, 
                                        Bool_t &wasopen) = 0;
  void SetParm(const char *parm, int val);
  void SetParm(const char *parm, double val);
};

#endif
