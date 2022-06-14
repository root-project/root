// @(#)root/graf:$Id$
// Author: Valeriy Onuchin   23/06/05

/*************************************************************************
 * Copyright (C) 2001-2002, Rene Brun, Fons Rademakers                   *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TASPluginGS
#define ROOT_TASPluginGS


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TASPluginGS                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TASImagePlugin.h"


class TASPluginGS : public TASImagePlugin {

private:
   char  *fInterpreter;   ///< path to GhostScript interpreter

public:
   TASPluginGS(const char *ext);
   virtual ~TASPluginGS();

   ASImage *File2ASImage(const char *filename) override;

   ClassDefOverride(TASPluginGS, 0)  // PS/EPS/PDF plugin based on GhostScript interpreter
};

#endif
