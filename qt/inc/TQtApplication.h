// @(#)root/qt:$Name:$:$Id:$
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
// TQtApplication                                                       //
//                                                                      //
// Interface to low level Qt package. This class gives access to basic  //
// Qt graphics, pixmap, text and font handling routines.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TQtApplicationThread.h"
#include "TQtRConfig.h"


class TQtApplication { // : public TApplicationImp

private:
  friend class TQtApplicationThread;
  TQtApplicationThread  *fGUIThread;

  void    CreateGUIThread(int argc, char **argv);

  static void CreateQApplication(int argc, char ** argv, bool GUIenabled);

protected:
   static TQtApplication *fgQtApplication;

public:

   TQtApplication() {};
   TQtApplication(const char *appClassName, int argc, char **argv);
   virtual ~TQtApplication();
   static bool Terminate();

   static TQtApplication *GetQtApplication();
   static bool IsThisGuiThread();
   // ClassDef(TQtApplication,0)

};
//______________________________________________________________________________
inline bool TQtApplication::IsThisGuiThread()
{
   // Check whether the current thread belongs the GUI
#ifdef R__QTGUITHREAD
 TQtApplication *app = GetQtApplication();
   if (!app) return TRUE;
   if (app->fGUIThread)
      return app->fGUIThread->IsThisThread();
#endif
  return TRUE;
}

#endif
