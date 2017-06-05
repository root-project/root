// @(#)root/base:$Id: TBuffer3D.cxx,v 1.00
// Author: Olivier Couet   05/05/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

/** \class TBuffer3D
\ingroup Base

Generic 3D primitive description class.
See TBuffer3DTypes for producer classes.

### Filling TBuffer3D and Adding to Viewer

The viewers behind the TVirtualViewer3D interface differ greatly in their
capabilities e.g.

  - Some know how to draw certain shapes natively (e.g. spheres/tubes in OpenGL)
  - others always require a raw tessellation description of points/lines/segments.
  - Some need the 3D object positions in the global frame, others can cope with
    local frames + a translation matrix - which can give considerable performance
    benefits.

To cope with these situations the object buffer is filled out in negotiation
with the viewer. TBuffer3D classes are conceptually divided into enumerated
sections Core, BoundingBox, Raw etc (see TBuffer3D.h for more details).

\image html base_tbuffer3d.png

The `SectionsValid() / SetSectionsValid / ClearSectionsValid()` methods of
TBuffer3D are used to test/set/clear these section valid flags.

The sections found in TBuffer3D (`Core/BoundingBox/Raw Sizes/Raw`) are sufficient
to describe any tessellated shape in a generic fashion. An additional
`ShapeSpecific` section in derived shape specific classes allows a more abstract
shape description ("a sphere of inner radius x, outer radius y"). This
enables a viewer which knows how to draw (tessellate) the shape itself to do so,
which can bring considerable performance and quality benefits, while providing a
generic fallback suitable for all viewers.

The rules for client negotiation with the viewer are:

  - If suitable specialized TBuffer3D class exists, use it, otherwise use TBuffer3D.
  - Complete the mandatory Core section.
  - Complete the ShapeSpecific section if applicable.
  - Complete the BoundingBox if you can.
  - Pass this buffer to the viewer using one of the AddObject() methods - see below.

If the viewer requires more sections to be completed (Raw/RawSizes) AddObject()
will return flags indicating which ones, otherwise it returns kNone. You must
fill the buffer and mark these sections valid, and pass the buffer again. A
typical code snippet would be:

~~~ {.cpp}
TBuffer3DSphere sphereBuffer;
// Fill out kCore...
// Fill out kBoundingBox...
// Fill out kShapeSpecific for TBuffer3DSphere
// Try first add to viewer
Int_t reqSections = viewer-&gt;AddObject(buffer);
if (reqSections != TBuffer3D::kNone) {
  if (reqSections & TBuffer3D::kRawSizes) {
     // Fill out kRawSizes...
  }
  if (reqSections & TBuffer3D::kRaw) {
     // Fill out kRaw...
  }
  // Add second time to viewer - ignore return cannot do more
  viewer-&gt;AddObject(buffer);
  }
}
~~~

`ShapeSpecific`: If the viewer can directly display the buffer without
filling of the kRaw/kRawSizes section it will not need to request client side
tessellation.

Currently we provide the following various shape specific classes, which the
OpenGL viewer can take advantage of (see TBuffer3D.h and TBuffer3DTypes.h)

  - TBuffer3DSphere - solid, hollow and cut spheres*
  - TBuffer3DTubeSeg - angle tube segment
  - TBuffer3DCutTube - angle tube segment with plane cut ends.


*OpenGL only supports solid spheres at present - cut/hollow ones will be
requested tessellated.

Anyone is free to add new TBuffer3D classes, but it should be clear that the
viewers require updating to be able to take advantage of them. The number of
native shapes in OpenGL will be expanded over time.

`BoundingBox:` You are not obliged to complete this, as any viewer
requiring one internally (OpenGL) will build one for you if you do not provide.
However to do this the viewer will force you to provide the raw tessellation, and the
resulting box will be axis aligned with the overall scene, which is non-ideal
for rotated shapes.

As we need to support orientated (rotated) bounding boxes, TBuffer3D requires
the 6 vertices of the box. We also provide a convenience function, SetAABoundingBox(),
for simpler case of setting an axis aligned bounding box.

### Master/Local Reference Frames

The `Core` section of TBuffer3D contains two members relating to reference
frames:
`fLocalFrame` & `fLocalMaster`. `fLocalFrame` indicates if any positions in the
buffer (bounding box and tessellation vertexes) are in local or master (world frame).
`fLocalMaster` is a standard 4x4 translation matrix (OpenGL column major ordering)
for placing the object into the 3D master frame.

If `fLocalFrame` is kFALSE, `fLocalMaster` should contain an identity matrix. This
is set by default, and can be reset using `SetLocalMasterIdentity()` function.

Logical & Physical Objects.
There are two cases of object addition:

  - Add this object as a single independent entity in the world reference frame.
  - Add a physical placement (copy) of this logical object (described in local
    reference frame).

The second case is very typical in geometry packages, GEANT4, where we have
very large number repeated placements of relatively few logical (unique) shapes.
Some viewers (OpenGL only at present) are able to take advantage of this by
identifying unique logical shapes from the `fID` logical ID member of
TBuffer3D. If repeated addition of the same `fID` is found, the shape
is cached already - and the costly tessellation does not need to be sent again.
The viewer can also perform internal GL specific caching with considerable
performance gains in these cases.

For this to work correctly the logical object in must be described in TBuffer3D
in the local reference frame, complete with the local/master translation. The
viewer indicates this through the interface method

~~~ {.cpp}
PreferLocalFrame()
~~~

If this returns kTRUE you can make repeated calls to AddObject(), with TBuffer3D
containing the same fID, and different `fLocalMaster` placements.

For viewers supporting logical/physical objects, the TBuffer3D content refers
to the properties of logical object, with the `fLocalMaster` transform and the
`fColor` and `fTransparency` attributes, which can be varied for each physical
object.

As a minimum requirement all clients must be capable of filling the raw tessellation
of the object buffer, in the master reference frame. Conversely viewers must
always be capable of displaying the object described by this buffer.

### Scene Rebuilds

It should be understood that AddObject is not an explicit command to the viewer
  - it may for various reasons decide to ignore it:

  - It already has the object internally cached .
  - The object falls outside some 'interest' limits of the viewer camera.
  - The object is too small to be worth drawing.

In all these cases AddObject() returns kNone, as it does for successful addition,
simply indicating it does not require you to provide further information about
this object. You should not try to make any assumptions about what the viewer
did with it.

This enables the viewer to be connected to a client which sends potentially
millions of objects, and only accept those that are of interest at a certain
time, caching the relatively small number of CPU/memory costly logical shapes,
and retaining/discarding the physical placements as required. The viewer may
decide to force the client to rebuild (republish) the scene (via
a TPad repaint at present), and thus collect these objects if the
internal viewer state changes. It does this presently by forcing a repaint
on the attached TPad object - hence the reason for putting all publishing to
the viewer in the attached pad objects Paint() method. We will likely remove
this requirement in the future, indicating the rebuild request via a normal
ROOT signal, which the client can detect.

### Physical IDs

TVirtualViewer3D provides for two methods of object addition:virtual Int_t AddObject(const
TBuffer3D & buffer, Bool_t * addChildren = 0)

~~~ {.cpp}
virtual Int_t AddObject(UInt_t physicalID, const TBuffer3D & buffer, Bool_t * addChildren = 0)
~~~

If you use the first (simple) case a viewer using logical/physical pairs
SetSectionsValid(TBuffer3D::kBoundingBox); will generate IDs for each physical
object internally. In the second you can specify a unique identifier from the
client, which allows the viewer to be more efficient. It can now cache both logical
and physical objects, and only discard physical objects no longer of interest as
part of scene rebuilds.

### Child Objects

In many geometries there is a rigid containment hierarchy, and so if the viewer
is not interested in a certain object due to limits/size then it will also
not be interest in any of the contained branch of descendents. Both AddObject()
methods have an addChildren parameter. The viewer will complete this (if passed)
indicating if children (contained within the one just sent) are worth adding.

### Recycling TBuffer3D

Once add AddObject() has been called, the contents are copied to the viewer
internally. You are free to destroy this object, or recycle it for the next
object if suitable.
*/

ClassImp(TBuffer3D);

////////////////////////////////////////////////////////////////////////////////
/// Destructor.
/// Construct from supplied shape type and raw sizes

TBuffer3D::TBuffer3D(Int_t type,
                     UInt_t reqPnts, UInt_t reqPntsCapacity,
                     UInt_t reqSegs, UInt_t reqSegsCapacity,
                     UInt_t reqPols, UInt_t reqPolsCapacity) :
      fType(type)
{
   Init();
   SetRawSizes(reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TBuffer3D::~TBuffer3D()
{
   if (fPnts) delete [] fPnts;
   if (fSegs) delete [] fSegs;
   if (fPols) delete [] fPols;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialise buffer.

void TBuffer3D::Init()
{
   fID            = 0;
   fColor         = 0;
   // Set fLocalMaster in section kCore to identity
   fTransparency  = 0;
   fLocalFrame    = kFALSE;
   fReflection    = kFALSE;
   SetLocalMasterIdentity();

   // Reset bounding box
   for (UInt_t v=0; v<8; v++) {
      for (UInt_t i=0; i<3; i++) {
         fBBVertex[v][i] = 0.0;
      }
   }
   // Set fLocalMaster in section kCore to identity

   // Set kRaw tessellation section of buffer with supplied sizes
   fPnts          = 0;
   fSegs          = 0;
   fPols          = 0;

   fNbPnts        = 0;
   fNbSegs        = 0;
   fNbPols        = 0;
   fPntsCapacity  = 0;
   fSegsCapacity  = 0;
   fPolsCapacity  = 0;
   // Set fLocalMaster in section kCore to identity

   // Wipe output section.
   fPhysicalID    = 0;

   // Set kRaw tessellation section of buffer with supplied sizes
   ClearSectionsValid();
}

////////////////////////////////////////////////////////////////////////////////
/// Clear any sections marked valid.

void TBuffer3D::ClearSectionsValid()
{
   fSections = 0U;
   SetRawSizes(0, 0, 0, 0, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Set kRaw tessellation section of buffer with supplied sizes.
/// Set fLocalMaster in section kCore to identity

void TBuffer3D::SetLocalMasterIdentity()
{
   for (UInt_t i=0; i<16; i++) {
      if (i%5) {
         fLocalMaster[i] = 0.0;
      }
      else {
         fLocalMaster[i] = 1.0;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set fBBVertex in kBoundingBox section to a axis aligned (local) BB
/// using supplied origin and box half lengths.
///~~~ {.cpp}
///   7-------6
///  /|      /|
/// 3-------2 |
/// | 4-----|-5
/// |/      |/
/// 0-------1
///~~~

void TBuffer3D::SetAABoundingBox(const Double_t origin[3], const Double_t halfLengths[3])
{
   // Vertex 0
   fBBVertex[0][0] = origin[0] - halfLengths[0];   // x
   fBBVertex[0][1] = origin[1] - halfLengths[1];   // y
   fBBVertex[0][2] = origin[2] - halfLengths[2];   // z
   // Vertex 1
   fBBVertex[1][0] = origin[0] + halfLengths[0];   // x
   fBBVertex[1][1] = origin[1] - halfLengths[1];   // y
   fBBVertex[1][2] = origin[2] - halfLengths[2];   // z
   // Vertex 2
   fBBVertex[2][0] = origin[0] + halfLengths[0];   // x
   fBBVertex[2][1] = origin[1] + halfLengths[1];   // y
   fBBVertex[2][2] = origin[2] - halfLengths[2];   // z
   // Vertex 3
   fBBVertex[3][0] = origin[0] - halfLengths[0];   // x
   fBBVertex[3][1] = origin[1] + halfLengths[1];   // y
   fBBVertex[3][2] = origin[2] - halfLengths[2];   // z
   // Vertex 4
   fBBVertex[4][0] = origin[0] - halfLengths[0];   // x
   fBBVertex[4][1] = origin[1] - halfLengths[1];   // y
   fBBVertex[4][2] = origin[2] + halfLengths[2];   // z
   // Vertex 5
   fBBVertex[5][0] = origin[0] + halfLengths[0];   // x
   fBBVertex[5][1] = origin[1] - halfLengths[1];   // y
   fBBVertex[5][2] = origin[2] + halfLengths[2];   // z
   // Vertex 6
   fBBVertex[6][0] = origin[0] + halfLengths[0];   // x
   fBBVertex[6][1] = origin[1] + halfLengths[1];   // y
   fBBVertex[6][2] = origin[2] + halfLengths[2];   // z
   // Vertex 7
   fBBVertex[7][0] = origin[0] - halfLengths[0];   // x
   fBBVertex[7][1] = origin[1] + halfLengths[1];   // y
   fBBVertex[7][2] = origin[2] + halfLengths[2];   // z
}

////////////////////////////////////////////////////////////////////////////////
/// Set kRaw tessellation section of buffer with supplied sizes.

Bool_t TBuffer3D::SetRawSizes(UInt_t reqPnts, UInt_t reqPntsCapacity,
                              UInt_t reqSegs, UInt_t reqSegsCapacity,
                              UInt_t reqPols, UInt_t reqPolsCapacity)
{
   Bool_t allocateOK = kTRUE;

   fNbPnts = reqPnts;
   fNbSegs = reqSegs;
   fNbPols = reqPols;

   if (reqPntsCapacity > fPntsCapacity) {
      delete [] fPnts;
      fPnts = new Double_t[reqPntsCapacity];
      if (fPnts) {
         fPntsCapacity = reqPntsCapacity;
      } else {
         fPntsCapacity = fNbPnts = 0;
         allocateOK = kFALSE;
      }
   }
   if (reqSegsCapacity > fSegsCapacity) {
      delete [] fSegs;
      fSegs = new Int_t[reqSegsCapacity];
      if (fSegs) {
         fSegsCapacity = reqSegsCapacity;
      } else {
         fSegsCapacity = fNbSegs = 0;
         allocateOK = kFALSE;
      }
   }
   if (reqPolsCapacity > fPolsCapacity) {
      delete [] fPols;
      fPols = new Int_t[reqPolsCapacity];
      if (fPols) {
         fPolsCapacity = reqPolsCapacity;
      } else {
         fPolsCapacity = fNbPols = 0;
         allocateOK = kFALSE;
      }
   }

   return allocateOK;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TBuffer3DSphere::TBuffer3DSphere(UInt_t reqPnts, UInt_t reqPntsCapacity,
                                 UInt_t reqSegs, UInt_t reqSegsCapacity,
                                 UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3D(TBuffer3DTypes::kSphere, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity),
   fRadiusInner(0.0), fRadiusOuter(0.0),
   fThetaMin(0.0), fThetaMax(180.0),
   fPhiMin(0.0), fPhiMax(360.0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Test if buffer represents a solid uncut sphere.

Bool_t TBuffer3DSphere::IsSolidUncut() const
{
   if (fRadiusInner   != 0.0   ||
       fThetaMin      != 0.0   ||
       fThetaMax      != 180.0 ||
       fPhiMin        != 0.0   ||
       fPhiMax        != 360.0 ) {
      return kFALSE;
   } else {
      return kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TBuffer3DTube::TBuffer3DTube(UInt_t reqPnts, UInt_t reqPntsCapacity,
                             UInt_t reqSegs, UInt_t reqSegsCapacity,
                             UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3D(TBuffer3DTypes::kTube, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity),
   fRadiusInner(0.0), fRadiusOuter(1.0), fHalfLength(1.0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TBuffer3DTube::TBuffer3DTube(Int_t type,
                             UInt_t reqPnts, UInt_t reqPntsCapacity,
                             UInt_t reqSegs, UInt_t reqSegsCapacity,
                             UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3D(type, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity),
   fRadiusInner(0.0), fRadiusOuter(1.0), fHalfLength(1.0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TBuffer3DTubeSeg::TBuffer3DTubeSeg(UInt_t reqPnts, UInt_t reqPntsCapacity,
                                   UInt_t reqSegs, UInt_t reqSegsCapacity,
                                   UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3DTube(TBuffer3DTypes::kTubeSeg, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity),
   fPhiMin(0.0), fPhiMax(360.0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TBuffer3DTubeSeg::TBuffer3DTubeSeg(Int_t type,
                                   UInt_t reqPnts, UInt_t reqPntsCapacity,
                                   UInt_t reqSegs, UInt_t reqSegsCapacity,
                                   UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3DTube(type, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity),
   fPhiMin(0.0), fPhiMax(360.0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TBuffer3DCutTube::TBuffer3DCutTube(UInt_t reqPnts, UInt_t reqPntsCapacity,
                                   UInt_t reqSegs, UInt_t reqSegsCapacity,
                                   UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3DTubeSeg(TBuffer3DTypes::kCutTube, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity)
{
   fLowPlaneNorm[0] = 0.0; fLowPlaneNorm[0] = 0.0; fLowPlaneNorm[0] = -1.0;
   fHighPlaneNorm[0] = 0.0; fHighPlaneNorm[0] = 0.0; fHighPlaneNorm[0] = 1.0;
}

//CS specific
UInt_t TBuffer3D::fgCSLevel = 0;

////////////////////////////////////////////////////////////////////////////////
/// Return CS level.

UInt_t TBuffer3D::GetCSLevel()
{
   return fgCSLevel;
}

////////////////////////////////////////////////////////////////////////////////
/// Increment CS level.

void TBuffer3D::IncCSLevel()
{
   ++fgCSLevel;
}

////////////////////////////////////////////////////////////////////////////////
/// Decrement CS level.

UInt_t TBuffer3D::DecCSLevel()
{
   return --fgCSLevel;
}
