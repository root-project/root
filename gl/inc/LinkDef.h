/* @(#)root/gl:$Name:  $:$Id: LinkDef.h,v 1.1.1.1 

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class TViewerOpenGL;
#pragma link C++ class TGLRenderArea;
#pragma link C++ class TGLWindow;
#pragma link C++ class TGLColorEditor;
#pragma link C++ class TGLGeometryEditor;
#pragma link C++ class TGLSceneEditor;
#pragma link C++ class TGLLightEditor;
#pragma link C++ class TArcBall;
#pragma link C++ class TGLCamera;
#pragma link C++ class TGLSceneObject;


#ifndef _WIN32
#pragma link C++ class TX11GL;
#endif

#endif
