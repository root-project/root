// @(#)root/base:$Name:  $:$Id: TVirtualViewer3D.h
// Author: Olivier Couet 05/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualViewer3D
#define ROOT_TVirtualViewer3D

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualViewer3D                                                     //
//                                                                      //
// Abstract 3D shapes viewer. The concrete implementations are:         //
//                                                                      //
// TViewerX3D   : X3d viewer                                            //
// TViewerOpenGL: OpenGL viewer                                         //
// TViewerPad3D : visualise the 3D scene in the current Pad             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TBuffer3D;
class TVirtualPad;

class TVirtualViewer3D
{
public:
   virtual ~TVirtualViewer3D() {};
   
   // Viewers must always handle master (absolute) positions - and
   // buffer producers must be able to supply them. Some viewers may 
   // prefer local frame & translation - and producers can optionally
   // supply them
   virtual Bool_t PreferLocalFrame() const = 0;  

   // Addition/removal of objects must occur between Begin/EndUpdate calls
   virtual void   BeginScene() = 0;
   virtual Bool_t BuildingScene() const = 0;
   virtual void   EndScene() = 0;

   // Simple object addition - buffer represents a unique single positioned object
   virtual Int_t  AddObject(const TBuffer3D & buffer, Bool_t * addChildren = 0) = 0;               

   // Complex object addition - for adding placed objects which have common template 
   // shapes. In this case buffer describes template shape (aside from kCore). 
   virtual Int_t  AddObject(UInt_t placedID, const TBuffer3D & buffer, Bool_t * addChildren = 0) = 0;
   
   virtual void   OpenComposite(const TBuffer3D & buffer, Bool_t * addChildren = 0) = 0;
   virtual void   CloseComposite() = 0;
   virtual void   AddCompositeOp(UInt_t operation) = 0;

   static  TVirtualViewer3D *Viewer3D(TVirtualPad *pad = 0, Option_t *type = "");

   ClassDef(TVirtualViewer3D,0) // Abstract interface to 3D viewers
};

#endif
