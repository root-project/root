// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEvePointSetProjectedGL
#define ROOT_TEvePointSetProjectedGL

#include <TPointSet3DGL.h>

class TGLViewer;
class TGLScene;


class TEvePointSetProjected;

class TEvePointSetProjectedGL : public TPointSet3DGL
{
private:
   TEvePointSetProjectedGL(const TEvePointSetProjectedGL&);            // Not implemented
   TEvePointSetProjectedGL& operator=(const TEvePointSetProjectedGL&); // Not implemented

protected:

public:
   TEvePointSetProjectedGL();
   virtual ~TEvePointSetProjectedGL();

   ClassDef(TEvePointSetProjectedGL, 0); // GL-renderer for TEvePointSetProjected class.
};

#endif
