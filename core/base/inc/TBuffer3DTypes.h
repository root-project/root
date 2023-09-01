// @(#)root/base:$Id$
// Author: Richard Maunder  10/3/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBuffer3DTypes
#define ROOT_TBuffer3DTypes

//TODO: Check on casing of enums - also clearer names would help

//TODO: Go through all shapes and check type is being set for each

// Scope to avoid clashes
class TBuffer3DTypes {
public:
                            // Buffer class        Producer class
                            //                     g3d              geom
   enum EType { kGeneric,   // TBuffer3D           Rest             Rest
                kComposite, // TBuffer3D                            TGetCompositeShape
                kLine,      // TBuffer3D           TPolyLine3D
                kMarker,    // TBuffer3D           TPolyMarker3D
                kSphere,    // TBuffer3DSphere     TSPHE            TGeoSphere
                kTube,      // TBuffer3DTube                        TGeoTube
                kTubeSeg,   // TBuffer3DTubeSeg                     TGeoTubeSeg
                kCutTube }; // TBuffer3DCutTube                     TGeoCtub
};

#endif
