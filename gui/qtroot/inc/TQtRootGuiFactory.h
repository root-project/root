// Author: Valeri Fine   13/05/2003
/****************************************************************************
** $Id: TQtRootGuiFactory.h,v 1.3 2007/11/02 17:08:10 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

#ifndef ROOT_TQtRootGuiFactory
#define ROOT_TQtRootGuiFactory

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQtRootGuiFactory                                                    //
//                                                                      //
// This class is a proxy-factory for Qt-base ROOT GUI components.       //
// It overrides the member functions of the X11/win32gdk-based          //
// TRootGuiFactory.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGuiFactory.h"

class TQtApplication;
class TVirtualX;
class TGClient;

class TQtRootGuiFactory : public  TGuiFactory {

private:
   TGuiFactory *fGuiProxy;
   
protected:
  static void CreateQClient();
  static TGClient *gfQtClient;

public:
   TQtRootGuiFactory();
   TQtRootGuiFactory(const char *name, const char *title= "Qt-based ROOT GUI Factory");
   virtual ~TQtRootGuiFactory() { delete fGuiProxy; }

   virtual TApplicationImp *CreateApplicationImp(const char *classname, int *argc, char **argv);

   virtual TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height);
   virtual TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);

   virtual TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height);
   virtual TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);

   virtual TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height, Option_t *opt);
   virtual TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *opt);

   virtual TContextMenuImp *CreateContextMenuImp(TContextMenu *c, const char *name, const char *title);

   virtual TControlBarImp *CreateControlBarImp(TControlBar *c, const char *title);
   virtual TControlBarImp *CreateControlBarImp(TControlBar *c, const char *title, Int_t x, Int_t y);

   virtual TInspectorImp *CreateInspectorImp(const TObject *obj, UInt_t width, UInt_t height);

   static TGClient *GetRootClient();
   ClassDef(TQtRootGuiFactory,0)  //Factory for Qt-based ROOT GUI components
};

inline TGClient *TQtRootGuiFactory::GetRootClient(){ return gfQtClient; }


#endif
