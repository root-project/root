// @(#)root/qt:$Id$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQtApplication
#define ROOT_TQtApplication

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQtApplication -  Instantiate the Qt system within ROOT environment  //
//                                                                      //
// Instantiate the Qt package by creating Qapplication object if any   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TQtRConfig.h"
#include "Rtypes.h"

class TQtApplicationThread;

class TQtApplication { // : public TApplicationImp
  
private:
  friend class TQtApplicationThread;
  TQtApplicationThread  *fGUIThread;

  void    CreateGUIThread(int &argc, char **argv);

  static void CreateQApplication(int &argc, char ** argv, bool GUIenabled);

  void operator=(const TQtApplication&);
  TQtApplication(const TQtApplication&);

protected:
   static TQtApplication *fgQtApplication;

public:

   TQtApplication() {fGUIThread=0;};
   TQtApplication(const char *appClassName, int &argc, char **argv);
   virtual ~TQtApplication();
   static bool Terminate();

   static TQtApplication *GetQtApplication();
   static bool IsThisGuiThread();
   static Int_t QtVersion();
   ClassDef(TQtApplication,0) // Instantiate the Qt system within ROOT environment

};
#endif
