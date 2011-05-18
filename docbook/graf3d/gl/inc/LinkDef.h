// @(#)root/gl:$Id$

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

#pragma link C++ class TGLWidget;
#pragma link C++ class TGLContext;
#pragma link C++ class TGLContextIdentity;
#pragma link C++ class TGLFormat;
#pragma link C++ class TGLFBO+;

#pragma link C++ class TGLVertex3;
#pragma link C++ class TGLVector3;
#pragma link C++ class TGLLine3;
#pragma link C++ class TGLRect;
#pragma link C++ class TGLPlane;
#pragma link C++ class TGLMatrix;
#pragma link C++ class TGLColor;
#pragma link C++ class TGLColorSet;
#pragma link C++ class TGLUtil;
#pragma link C++ class TGLUtil::TColorLocker;
#pragma link C++ class TGLUtil::TDrawQualityModifier;
#pragma link C++ class TGLUtil::TDrawQualityScaler;
#pragma link C++ class TGLQuadric;

#pragma link C++ class TGLStopwatch;
#pragma link C++ class TGLLockable;
#pragma link C++ class TGLBoundingBox;
#pragma link C++ class TGLRnrCtx+;
#pragma link C++ class TGLSelectBuffer+;
#pragma link C++ class TGLSelectRecordBase+;
#pragma link C++ class TGLSelectRecord+;
#pragma link C++ class TGLOvlSelectRecord+;

#pragma link C++ class TGLLogicalShape;
#pragma link C++ class TGLPhysicalShape;

#pragma link C++ class TGLClip+;
#pragma link C++ class TGLClipPlane+;
#pragma link C++ class TGLClipBox+;
#pragma link C++ class TGLClipSet+;
#pragma link C++ class TGLClipSetEditor+;
#pragma link C++ class TGLClipSetSubEditor+;

#pragma link C++ class TGLManip;
#pragma link C++ class TGLScaleManip;
#pragma link C++ class TGLTransManip;
#pragma link C++ class TGLRotateManip;
#pragma link C++ class TGLManipSet;

#pragma link C++ class TGLCamera;
#pragma link C++ class TGLOrthoCamera;
#pragma link C++ class TGLPerspectiveCamera;
#pragma link C++ class TGLCameraOverlay;
#pragma link C++ class TGLCameraGuide;
#pragma link C++ class TGLPlotCamera+;
#pragma link C++ class TGLAutoRotator+;

#pragma link C++ class TGLSceneBase+;
#pragma link C++ class TGLScene+;
#pragma link C++ class TGLScenePad+;
#pragma link C++ class TGLSceneInfo+;
#pragma link C++ class TGLScene::TSceneInfo+;
#pragma link C++ class TGLOverlayElement+;
#pragma link C++ class TGLOverlayList+;
#pragma link C++ class TGLOverlayButton+;
#pragma link C++ class TGLAnnotation+;

#pragma link C++ class TGLViewerBase+;
#pragma link C++ class TGLViewer+;
#pragma link C++ class TGLEventHandler;
#pragma link C++ class TGLFaderHelper+;
#pragma link C++ class TGLViewerEditor+;
#pragma link C++ class TGLEmbeddedViewer;
#pragma link C++ class TGLSAViewer;
#pragma link C++ class TGLSAFrame;

#pragma link C++ class TGLPShapeRef+;
#pragma link C++ class TGLPShapeObj+;
#pragma link C++ class TGLPShapeObjEditor+;

#pragma link C++ class TGLLightSet+;
#pragma link C++ class TGLLightSetEditor+;
#pragma link C++ class TGLLightSetSubEditor+;


#pragma link C++ class TGLOutput;

#pragma link C++ class TArcBall;

#pragma link C++ class TGLFaceSet;
#pragma link C++ class TGLPolyLine;
#pragma link C++ class TGLPolyMarker;
#pragma link C++ class TGLCylinder;
#pragma link C++ class TGLSphere;
#pragma link C++ class TGLText;
#pragma link C++ class TGLAxis;
#pragma link C++ class TGLAxisPainter+;
#pragma link C++ class TGLAxisPainterBox+;

#pragma link C++ class TGLSelectionBuffer;
#pragma link C++ class TGLPlotCoordinates;
#pragma link C++ class TGLSurfacePainter;
#pragma link C++ class TGLVoxelPainter;
#pragma link C++ class TGLHistPainter;
#pragma link C++ class TGLLegoPainter;
#pragma link C++ class TGLPlotPainter;
#pragma link C++ class TGLBoxPainter;
#pragma link C++ class TGLTF3Painter;
#pragma link C++ class TGLIsoPainter;
#pragma link C++ class TGLPlotBox;
#pragma link C++ class TGLTH3Slice;
#pragma link C++ class TGLBoxCut;
#pragma link C++ class TGLParametricEquation;
#pragma link C++ class TGLParametricPlot;
#pragma link C++ class TGLAdapter;
#pragma link C++ class TF2GL+;
#pragma link C++ class TH2GL+;
#pragma link C++ class TH3GL+;
#pragma link C++ class TGLH2PolyPainter;
#pragma link C++ class TGLParametricEquationGL;
#pragma link C++ class TGLPadPainter;
#pragma link C++ class TGL5DDataSet;
#pragma link C++ class TGL5DDataSetEditor;
#pragma link C++ class TGLTH3Composition;
#pragma link C++ class TGLTH3CompositionPainter;

#ifndef _WIN32
#pragma link C++ class TX11GLManager;
#endif

#pragma link C++ class TGLObject+;
#pragma link C++ class TGLPlot3D+;
#pragma link C++ class TPointSet3DGL+;

#pragma link C++ class TGLFont;
#pragma link C++ class TGLFontManager;

#pragma link C++ namespace Rgl;

#endif
