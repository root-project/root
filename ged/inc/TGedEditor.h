// @(#)root/ged:$Name:  $:$Id: TGedEditor.h,v 1.0 2003/12/02 13:41:59 rdm Exp $
// Author: Marek Biskup, Ilka Antcheva   02/12/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedEditor
#define ROOT_TGedEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedEditor (very first prototype)                                    //
//                                                                      //
// Editor is a composite frame that contains ToolBox and TGedAttFrames. //
// It connects to a Canvas and listens for selected objects             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TCanvas;
class TGedAttFrame;
class TGedToolBox;
class TGedPropertyFrame;


class TGedEditor : public TGMainFrame {

protected:
   TGedToolBox           *fToolBox;
   TGedPropertyFrame *fPropertiesFrame;

   virtual void Build();

public:
   TGedEditor(TCanvas* canvas = 0);
   virtual ~TGedEditor();

   virtual void ConnectToCanvas(TCanvas *c);
   
   ClassDef(TGedEditor,0)  //new editor (very first prototype)
};

#endif
