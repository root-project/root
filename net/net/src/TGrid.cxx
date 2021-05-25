// @(#)root/net:$Id$
// Author: Fons Rademakers   3/1/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGrid                                                                //
//                                                                      //
// Abstract base class defining interface to common GRID services.      //
//                                                                      //
// To open a connection to a GRID use the static method Connect().      //
// The argument of Connect() is of the form:                            //
//    <grid>[://<host>][:<port>], e.g. alien://alice.cern.ch            //
// Depending on the <grid> specified an appropriate plugin library      //
// will be loaded which will provide the real interface.                //
//                                                                      //
// Related classes are TGridResult.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGrid.h"
#include "TROOT.h"
#include "TPluginManager.h"
#include "TError.h"

TGrid *gGrid = 0;


ClassImp(TGrid);

////////////////////////////////////////////////////////////////////////////////
/// The grid should be of the form:  <grid>://<host>[:<port>],
/// e.g.:  alien://alice.cern.ch
/// The uid is the username and pw the password that should be used for
/// the connection. Depending on the <grid> the shared library (plugin)
/// for the selected system will be loaded. When the connection could not
/// be opened 0 is returned. For AliEn the supported options are:
/// -domain=<domain name>
/// -debug=<debug level from 1 to 10>
/// Example: "-domain=cern.ch -debug=5"

TGrid *TGrid::Connect(const char *grid, const char *uid, const char *pw,
                      const char *options)
{
   TPluginHandler *h;
   TGrid *g = 0;

   if (!grid) {
      ::Error("TGrid::Connect", "no grid specified");
      return 0;
   }
   if (!uid)
      uid = "";
   if (!pw)
      pw = "";
   if (!options)
      options = "";

   if ((h = gROOT->GetPluginManager()->FindHandler("TGrid", grid))) {
      if (h->LoadPlugin() == -1)
         return 0;
      g = (TGrid *) h->ExecPlugin(4, grid, uid, pw, options);
   }

   return g;
}
