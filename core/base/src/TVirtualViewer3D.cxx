// @(#)root/base:$Id$
// Author: Olivier Couet 05/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualViewer3D                                                     //
//                                                                      //
// Abstract 3D shapes viewer. The concrete implementations are:         //
//                                                                      //
// TViewerX3D   : X3d viewer                                            //
// TGLViewer    : OpenGL viewer                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//BEGIN_HTML <!--
/* -->
<h4>3D Viewer Infrastructure Overview</h4>
<p>The 3D Viewer infrastructure consists of:</p>
<ul>
  <li> TVirtualViewer3D interface: An abstract handle to the viewer, allowing
    client to test preferences, add objects, control the viewer via scripting
    (to be added) etc.</li>
  <li>TBuffer3D class hierarchy: Used to describe 3D objects
    (&quot;shapes&quot;)
    - filled /added by negotiation with viewer via TVirtualViewer3D.</li>
</ul>
<p>Together these allow clients to publish objects to any one of the 3D viewers
  (currently OpenGL/x3d,TPad), free of viewer specific drawing code. They allow
  our simple x3d viewer, and considerably more sophisticated OpenGL one to both
  work with both geometry libraries (g3d and geom) efficiently.</p>
<p>Publishing to a viewer consists of the following steps:</p>
<ol>
  <li> Create / obtain viewer handle</li>
  <li>Begin scene on viewer</li>
  <li>Fill mandatory parts of TBuffer3D describing object</li>
  <li>Add to viewer</li>
  <li>Fill optional parts of TBuffer3D if requested by viewer, and add again<br>
     ... repeat 3/4/5
  as required</li>
  <li>End scene on viewer</li>
</ol>
<h4>Creating / Obtaining Viewer</h4>
<p>Create/obtain the viewer handle via local/global pad - the viewer is always
  bound to a TPad object at present [This may be removed as a restriction in
  the future] . You should perform the publishing to the viewer described below
  in the Paint() method of the object you attach to the pad (via Draw())</p>
<pre>TVirtualViewer3D * v = gPad-&gt;GetViewer3D(&quot;xxxx&quot;);</pre>
<p>&quot; xxxx&quot; is viewer type: OpenGL &quot;ogl&quot;, X3D &quot;x3d&quot; or
  Pad &quot;pad&quot; (default). The viewer is created via the plugin manager,
  attached to pad, and the interface returned.</p>
<h4> Begin / End Scene</h4>
<p>Objects must be added to viewer between BeginScene/EndScene calls e.g.</p>
<pre>v-&gt;BeginScene();
.....
v-&gt;AddObject(....);
v-&gt;AddObject(....);
.....
v-&gt;EndScene();</pre>
<p>The BeginScene call will cause the viewer to suspend redraws etc, and after
  the EndScene the viewer will reset the camera to frame the new scene and redraw.</p>
[x3d viewer does not support changing of scenes - objects added after the
  first Open/CloseScene pair will be ignored.]<br>
<h4>Filling TBuffer3D and Adding to Viewer</h4>
<p>The viewers behind the TVirtualViewer3D interface differ greatly in their
  capabilities e.g.</p>
<ul>
  <li> Some know how to draw certain shapes natively (e.g. spheres/tubes in
    OpenGL) - others always require a raw tessellation description of points/lines/segments.</li>
  <li>Some
      need the 3D object positions in the global frame, others can cope with
    local frames + a translation matrix - which can give considerable performance
      benefits.</li>
</ul>
<p>To cope with these situations the object buffer is filled out in negotiation
  with the viewer. TBuffer3D classes are conceptually divided into enumerated
  sections Core, BoundingBox, Raw etc (see TBuffer3D.h for more details). </p>
<p align="center"><img src="gif/TBuffer3D.gif" width="501" height="501"></p>
<p>The<em> SectionsValid() / SetSectionsValid / ClearSectionsValid() </em>methods of TBuffer3D
    are used to test/set/clear these section valid flags.</p>
<p>The sections found in TBuffer3D (<em>Core/BoundingBox/Raw Sizes/Raw</em>)
  are sufficient to describe any tessellated shape in a generic fashion. An additional <em>ShapeSpecific</em>  section
  in derived shape specific classes allows a more abstract shape description
  (&quot;a sphere of inner radius x, outer radius y&quot;). This enables a viewer
  which knows how to draw (tessellate) the shape itself to do so, which can bring
  considerable performance and quality benefits, while providing a generic fallback
  suitable for all viewers.</p>
<p>The rules for client negotiation with the viewer are:</p>
<ul>
  <li> If suitable specialized TBuffer3D class exists, use it, otherwise use
    TBuffer3D.</li>
  <li>Complete the mandatory Core section.</li>
  <li>Complete the ShapeSpecific section
      if applicable.</li>
  <li>Complete the BoundingBox if you can.</li>
  <li>Pass this buffer to the viewer using
      one of the AddObject() methods - see below.</li>
</ul>
<p>If the viewer requires more sections to be completed (Raw/RawSizes) AddObject()
  will return flags indicating which ones, otherwise it returns kNone. You must
  fill the buffer and mark these sections valid, and pass the buffer again. A
  typical code snippet would be:</p>
<pre>TBuffer3DSphere sphereBuffer;
// Fill out kCore...
// Fill out kBoundingBox...
// Fill out kShapeSpecific for TBuffer3DSphere
// Try first add to viewer
Int_t reqSections = viewer-&gt;AddObject(buffer);
if (reqSections != TBuffer3D::kNone) {
  if (reqSections &amp; TBuffer3D::kRawSizes) {
     // Fill out kRawSizes...
  }
  if (reqSections &amp; TBuffer3D::kRaw) {
     // Fill out kRaw...
  }
  // Add second time to viewer - ignore return cannot do more
  viewer-&gt;AddObject(buffer);
  }
}</pre>
<p><em>ShapeSpecific</em>: If the viewer can directly display the buffer without
  filling of the kRaw/kRawSizes section it will not need to request client side
  tessellation.
  Currently we provide the following various shape specific classes, which the
  OpenGL viewer can take advantage of (see TBuffer3D.h and TBuffer3DTypes.h)</p>
<ul>
  <li>TBuffer3DSphere - solid, hollow and cut spheres*</li>
  <li>TBuffer3DTubeSeg - angle tube segment</li>
  <li>TBuffer3DCutTube - angle tube segment with plane cut ends.</li>
</ul>
<p>*OpenGL only supports solid spheres at present - cut/hollow ones will be
    requested tessellated.</p>
<p>Anyone is free to add new TBuffer3D classes, but it should be clear that the
  viewers require updating to be able to take advantage of them. The number of
  native shapes in OpenGL will be expanded over time.</p>
<p><em>BoundingBox: </em>You are not obliged to complete this, as any viewer
  requiring one internally (OpenGL) will build one for you if you do not provide.
  However
  to do this the viewer will force you to provide the raw tessellation, and the
  resulting box will be axis aligned with the overall scene, which is non-ideal
  for rotated shapes.</p>
<p>As we need to support orientated (rotated) bounding boxes, TBuffer3D requires
  the 6 vertices of the box. We also provide a convenience function, SetAABoundingBox(),
  for simpler case of setting an axis aligned bounding box.</p>
<h4>
  Master/Local Reference Frames</h4>
The <em>Core</em> section of TBuffer3D contains two members relating to reference
  frames:
<em>fLocalFrame</em> &amp; <em>fLocalMaster</em>. <em>fLocalFrame</em> indicates
  if any positions in the buffer (bounding box and tessellation vertexes) are
  in local or master (world
  frame). <em>fLocalMaster</em> is a standard 4x4 translation matrix (OpenGL
  colum major ordering) for placing the object into the 3D master frame.
  <p>If <em>fLocalFrame</em> is kFALSE, <em>fLocalMaster</em> should contain an
  identity matrix. This is set by default, and can be reset using <em>SetLocalMasterIdentity()</em> function.<br>
Logical &amp; Physical Objects</p>
<p>There are two cases of object addition:</p>
<ul>
  <li> Add this object as a single independent entity in the world reference
    frame.</li>
  <li>Add
        a physical placement (copy) of this logical object (described in local
    reference frame).</li>
</ul>
<p>The second case is very typical in geometry packages, GEANT4, where we have
  very large number repeated placements of relatively few logical (unique) shapes.
  Some viewers (OpenGL only at present) are able to take advantage of this by
  identifying unique logical shapes from the <em>fID</em> logical ID member of
  TBuffer3D. If repeated addition of the same <em>fID</em> is found, the shape
  is cached already - and the costly tessellation does not need to be sent again.
  The viewer can
  also perform internal GL specific caching with considerable performance gains
  in these cases.</p>
<p>For this to work correctly the logical object in must be described in TBuffer3D
  in the local reference frame, complete with the local/master translation. The
  viewer indicates this through the interface method</p>
<pre>PreferLocalFrame()</pre>
<p>If this returns kTRUE you can make repeated calls to AddObject(), with TBuffer3D
  containing the same fID, and different <em>fLocalMaster</em> placements.</p>
<p>For viewers supporting logical/physical objects, the TBuffer3D content refers
  to the properties of logical object, with the <em>fLocalMaster</em> transform and the
  <em>fColor</em> and <em>fTransparency</em> attributes, which can be varied for each physical
  object.</p>
<p>As a minimum requirement all clients must be capable of filling the raw tessellation
  of the object buffer, in the master reference frame. Conversely viewers must
  always be capable of displaying the object described by this buffer.</p>
<h4>
  Scene Rebuilds</h4>
<p>It should be understood that AddObject is not an explicit command to the viewer
  - it may for various reasons decide to ignore it:</p>
<ul>
  <li> It already has the object internally cached .</li>
  <li>The object falls outside
    some 'interest' limits of the viewer camera.</li>
  <li>The object is too small to
      be worth drawing.</li>
</ul>
<p>In all these cases AddObject() returns kNone, as it does for successful addition,
  simply indicating it does not require you to provide further information about
  this object. You should
  not try to make any assumptions about what the viewer did with it.</p>
<p>This enables the viewer to be connected to a client which sends potentially
  millions of objects, and only accept those that are of interest at a certain
  time, caching the relatively small number of CPU/memory costly logical shapes,
  and retaining/discarding the physical placements as required. The viewer may
  decide to force the client to rebuild (republish) the scene (via
  a TPad
  repaint
  at
  present),
  and
  thus
  collect
  these
  objects if
  the
  internal viewer state changes. It does this presently by forcing a repaint
  on the attached TPad object - hence the reason for putting all publishing to
  the viewer in the attached pad objects Paint() method. We will likely remove
  this requirement in the future, indicating the rebuild request via a normal
ROOT signal, which the client can detect.</p>
<h4>
  Physical IDs</h4>
TVirtualViewer3D provides for two methods of object addition:virtual Int_t AddObject(const
TBuffer3D &amp; buffer, Bool_t * addChildren = 0)<br>
<pre>virtual Int_t AddObject(UInt_t physicalID, const TBuffer3D &amp; buffer, Bool_t * addChildren = 0)</pre>
<p>If you use the first (simple) case a viewer using logical/physical pairs
    will generate IDs for each physical object internally. In the second you
    can specify
      a unique identifier from the client, which allows the viewer to be more
    efficient. It can now cache both logical and physical objects, and only discard
    physical
  objects no longer of interest as part of scene rebuilds.</p>
<h4>
  Child Objects</h4>
<p>In many geometries there is a rigid containment hierarchy, and so if the viewer
  is not interested in a certain object due to limits/size then it will also
  not be interest in any of the contained branch of descendents. Both AddObject()
  methods have an addChildren parameter. The viewer will complete this (if passed)
indicating if children (contained within the one just sent) are worth adding.</p>
<h4>
  Recyling TBuffer3D </h4>
<p>Once add AddObject() has been called, the contents are copied to the viewer
  internally. You are free to destroy this object, or recycle it for the next
  object if suitable.</p>
<!--*/
// -->END_HTML


#include "TVirtualViewer3D.h"
#include "TVirtualPad.h"
#include "TPluginManager.h"
#include "TError.h"
#include "TClass.h"

ClassImp(TVirtualViewer3D)

//______________________________________________________________________________
TVirtualViewer3D* TVirtualViewer3D::Viewer3D(TVirtualPad *pad, Option_t *type)
{
   // Create a Viewer 3D of specified type.

   TVirtualViewer3D *viewer = 0;
   TPluginHandler *h;
   if ((h = gPluginMgr->FindHandler("TVirtualViewer3D", type))) {
      if (h->LoadPlugin() == -1)
         return 0;

      if (!pad) {
         viewer = (TVirtualViewer3D *) h->ExecPlugin(1, gPad);
      } else {
         viewer = (TVirtualViewer3D *) h->ExecPlugin(1, pad);
      }
   }
   return viewer;
}
