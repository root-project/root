// @(#)root/ged:$Name:  $:$Id: TGedPadEditor.h,v 1.0 2003/12/02 13:41:59 rdm Exp $
// Author: Marek Biskup, Ilka Antcheva   02/12/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedPadEditor
#define ROOT_TGedPadEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedPadEditor                                                        //
//                                                                      //
// Editor contains ToolBox and TGedAttFrames.                           //
// It is connected to a canvas and listens for selected objects         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualPadEditor
#include "TVirtualPadEditor.h"
#endif

class TCanvas;
class TGedEditor;

class TGedPadEditor : public TVirtualPadEditor {

protected:
   TGedEditor  *fEdit;  // new pad editor

public:
   TGedPadEditor(TCanvas* canvas = 0);
   virtual ~TGedPadEditor();

   TGedEditor *GetEditor() const { return fEdit; }

   ClassDef(TGedPadEditor,0)  //new editor
};

#endif
