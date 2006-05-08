// @(#)root/gl:$Name:  $:$Id: TGLDrawFlags.h,v 1.2 2006/02/09 09:56:20 couet Exp $
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

public:
   TGLDrawFlags(EStyle style = kFill, Short_t LOD = kLODHigh);
   virtual ~TGLDrawFlags();

   EStyle  Style() const          { return fStyle; }
   void    SetStyle(EStyle style) { fStyle = style; }
   Short_t LOD() const            { return fLOD; }
   void    SetLOD(Short_t LOD)    { fLOD = LOD; }

   ClassDef(TGLDrawFlags,0) // GL draw flags wrapper class
};

#endif // ROOT_TGLDrawFlags
