// @(#)root/ged:$Name:  $:$Id: TGedPadEditor.cxx,v 1.0 2003/12/02 13:41:59 rdm Exp $
// Author: Ilka Antcheva 17/02/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedPadEditor                                                        //
//                                                                      //
// It creates a TGedEditor to be used as a new pad editor               //
// (a prototype)                                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGedPadEditor.h"
#include "TGedEditor.h"
#include "TCanvas.h"

ClassImp(TGedPadEditor)

//______________________________________________________________________________
TGedPadEditor::TGedPadEditor(TCanvas* canvas) :
   TVirtualPadEditor()
{
   // Constructor
   
   fEdit = new TGedEditor(canvas);
}

//______________________________________________________________________________
TGedPadEditor::~TGedPadEditor()
{
   // Destructor

   delete fEdit;
}
