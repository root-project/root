// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TQtApplication.h,v 1.7 2004/06/28 20:16:54 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

#ifndef ROOT_TQtApplication
#define ROOT_TQtApplication

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGQt                                                                  //
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
