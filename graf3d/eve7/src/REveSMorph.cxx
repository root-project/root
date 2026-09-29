// @(#)root/eve7:$Id$
// Author: Matevz Tadel

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveSMorph.hxx>
#include <ROOT/REveRenderData.hxx>
#include <ROOT/REveTrans.hxx>

#include <cmath>

using namespace ROOT::Experimental;

/** \class REveSMorph
\ingroup REve
A parametric, texture-mapped surface of spherical topology: a sphere that can
be twisted, pinched and sheared, and cut down to a patch in theta and phi.
Ported from Gled, whose parameter names it keeps.

The geometry is built on the client, by `makeSMorph` in EveElementsRCore.js.
The surface is textured and `REveRenderData` has no channel for texture
coordinates, so only the parameters are streamed.

The surface is generated at unit size. Radius, position and orientation are
in the element's transformation, so changing them streams one matrix under
kCBTransBBox and the client rebuilds nothing. Every other setter stamps
kCBObjProps and forces a rebuild.

The parametrisation, with ct = cos(theta), st = sin(theta):

    twist = ct * fTx,  conv = ct * fCx
    x = ct
    y = (1 + conv) * st * cos(phi + twist)
    z = (1 + conv) * st * sin(phi + twist)

followed by a rotation about z through x * fRz, which shears the body along
its polar axis. The polar axis is x, as in the original. The normal is the
normalised position, which is exact for the unmorphed sphere and approximate
for small fTx, fCx and fRz.

Texture coordinates:

    u = fTexX0 + fTexXC * phi / 2pi
    v = fTexY0 + fTexYC * acos(ct) / pi

A non-zero fTexYOff adds int(v) * fTexYOff to u, which offsets successive
wraps into a brick bond.

Worked example: `tutorials/visualisation/eve7/boing.C`.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor. Points the main colour at fColor.

REveSMorph::REveSMorph(const std::string &n, const std::string &t) : REveElement(n, t)
{
   SetMainColorPtr(&fColor);
}

////////////////////////////////////////////////////////////////////////////////
/// Uniform scale on the main transformation. Stamps kCBTransBBox only: the
/// surface is built at unit size, so the client rebuilds nothing.

void REveSMorph::SetRadius(Float_t r)
{
   RefMainTrans().SetScale(r, r, r);
   StampTransBBox();
}

////////////////////////////////////////////////////////////////////////////////
/// Stream the shape and texture parameters and the bounding box.

Int_t REveSMorph::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);

   j["fTLevel"]   = fTLevel;
   j["fPLevel"]   = fPLevel;

   j["fTx"]       = fTx;
   j["fCx"]       = fCx;
   j["fRz"]       = fRz;

   j["fThetaMin"] = fThetaMin;
   j["fThetaMax"] = fThetaMax;
   j["fPhiMean"]  = fPhiMean;
   j["fPhiRange"] = fPhiRange;
   j["fEquiSurf"] = fEquiSurf;

   j["fTexture"]  = fTexture;
   j["fTexX0"]    = fTexX0;
   j["fTexY0"]    = fTexY0;
   j["fTexXC"]    = fTexXC;
   j["fTexYC"]    = fTexYC;
   j["fTexYOff"]  = fTexYOff;

   // Bounding box as JSON, min triple then max triple, for RC.Box3. The client
   // could measure its own geometry; the box is sent because it is known here.
   ComputeBBox();
   const Float_t *bb = GetBBox();
   j["bbox"] = {bb[0], bb[2], bb[4], bb[1], bb[3], bb[5]};

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Bounding box: the unit cube in the element's own frame. It does not follow
/// the morph, so fCx > 0 can push the surface outside it. Tracking the morph
/// would rescale the 3D axis, which takes its extent from the scene box, while
/// the fCx slider is dragged.

void REveSMorph::ComputeBBox()
{
   BBoxInit();
   BBoxCheckPoint(-1.f, -1.f, -1.f);
   BBoxCheckPoint( 1.f,  1.f,  1.f);
}

////////////////////////////////////////////////////////////////////////////////
/// Name the client factory makeSMorph. No geometry is written; the base call
/// adds the transformation, and one vertex keeps the buffer non-empty.

void REveSMorph::BuildRenderData()
{
   fRenderData = std::make_unique<REveRenderData>("makeSMorph");
   REveElement::BuildRenderData();
   fRenderData->PushV(0.f, 0.f, 0.f);
}
