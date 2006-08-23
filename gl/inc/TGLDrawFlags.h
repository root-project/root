// @(#)root/gl:$Name:  $:$Id: TGLDrawFlags.h,v 1.3 2006/05/08 14:01:08 rdm Exp $
// Author:  Richard Maunder  27/01/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLDrawFlags
#define ROOT_TGLDrawFlags

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLDrawFlags                                                         //      
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGLDrawFlags
{
public:
   enum EStyle { kFill, kOutline, kWireFrame };
   enum ELODPresets {
      kLODPixel       = 0, // Projected size pixel or less
      kLODLow         = 20,
      kLODMed         = 50,
      kLODHigh        = 100,
      kLODUnsupported = 200 // Used to draw/DL cache drawables with LODSupport() of TGLDrawable::kLODAxesNone
   };

private:
   // Fields
   EStyle  fStyle;
   Short_t fLOD;
   Bool_t  fSelection;
   Bool_t  fSecSelection;

public:
   TGLDrawFlags(EStyle style = kFill, Short_t LOD = kLODHigh,
                Bool_t sel = kFALSE, Bool_t secSel = kFALSE);
   virtual ~TGLDrawFlags();

   EStyle  Style() const          { return fStyle; }
   void    SetStyle(EStyle style) { fStyle = style; }
   Short_t LOD() const            { return fLOD; }
   void    SetLOD(Short_t LOD)    { fLOD = LOD; }
   Bool_t  Selection() const              { return fSelection; }
   void    SetSelection(Bool_t sel)       { fSelection = sel; }
   Bool_t  SecSelection() const           { return fSecSelection; }
   void    SetSecSelection(Bool_t secSel) { fSecSelection = secSel; }

   ClassDef(TGLDrawFlags,0) // GL draw flags wrapper class
};

#endif // ROOT_TGLDrawFlags
