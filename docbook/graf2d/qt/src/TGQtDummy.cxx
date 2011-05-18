// @(#)root/qt:$Id$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGQtDummy                                                            //
//                                                                      //
// Dummy implementations of the Qt graphics system interface.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGQt.h"

   Window_t     TGQt::GetPrimarySelectionOwner(){return 0;}
   void         TGQt::SetPrimarySelectionOwner(Window_t /*id*/){;}
   void         TGQt::ConvertPrimarySelection(Window_t /*id*/, Atom_t /*clipboard*/, Time_t /*when*/){;}

   void         TGQt::FreeFontStruct(FontStruct_t /*fs*/){;}

   void         TGQt::FreeColor(Colormap_t /*cmap*/, ULong_t /*pixel*/){;}

   Int_t        TGQt::AddWindow(ULong_t /*qwid*/, UInt_t /*w*/, UInt_t /*h*/){return 0;}
   void         TGQt::RemoveWindow(ULong_t /*qwid*/){;}

   Handle_t     TGQt::GetNativeEvent() const { return 0; }

#if ROOT_VERSION_CODE < ROOT_VERSION(4,01,01)
// -------------------  OpenGL interface ------------------------
   Window_t     TGQt::CreateGLWindow(Window_t      /*wind*/, Visual_t /*visual*/, Int_t /*depth*/){return 0;}
   ULong_t      TGQt::wglCreateContext(Window_t    /*wind*/){return 0;}
   void         TGQt::wglDeleteContext(ULong_t     /*ctx*/){}
   void         TGQt::wglMakeCurrent(Window_t      /*wind*/, ULong_t  /*ctx*/){}
   void         TGQt::wglSwapLayerBuffers(Window_t /*wind*/, UInt_t  /*mode*/){}
   void         TGQt::glViewport(Int_t  /*x0*/, Int_t  /*y0*/, Int_t  /*x1*/, Int_t  /*y1*/){}
   void         TGQt::glClearIndex(Float_t  /*fParam*/){}
   void         TGQt::glClearColor(Float_t  /*red*/, Float_t  /*green*/, Float_t  /*blue*/, Float_t  /*alpha*/){}
   void         TGQt::glDrawBuffer(UInt_t  /*mode*/){}
   void         TGQt::glClear(UInt_t  /*mode*/){}
   void         TGQt::glDisable(UInt_t  /*mode*/){}
   void         TGQt::glEnable(UInt_t  /*mode*/){}
   void         TGQt::glFlush(){}
   void         TGQt::glFrontFace(UInt_t  /*mode*/){}
   void         TGQt::glNewList(UInt_t  /*list*/, UInt_t  /*mode*/){}
   void         TGQt::glGetBooleanv(UInt_t  /*mode*/, UChar_t  * /*bRet*/){}
   void         TGQt::glGetDoublev(UInt_t  /*mode*/, Double_t  * /*dRet*/){}
   void         TGQt::glGetFloatv(UInt_t  /*mode*/, Float_t  * /*fRet*/){}
   void         TGQt::glGetIntegerv(UInt_t  /*mode*/, Int_t  * /*iRet*/){}
   Int_t        TGQt::glGetError(){return -1;}
   void         TGQt::glEndList(){}
   void         TGQt::glBegin(UInt_t  /*mode*/){}
   void         TGQt::glEnd(){}
   void         TGQt::glPushMatrix(){}
   void         TGQt::glPopMatrix(){}
   void         TGQt::glRotated(Double_t  /*angle*/, Double_t  /*x*/, Double_t  /*y*/, Double_t  /*z*/){}
   void         TGQt::glTranslated(Double_t  /*x*/, Double_t  /*y*/, Double_t  /*z*/){}
   void         TGQt::glMultMatrixd(const Double_t  * /*matrix*/){}
   void         TGQt::glColor3fv(const Float_t  * /*color*/){}
   void         TGQt::glVertex3f(Float_t  /*x*/, Float_t  /*y*/, Float_t  /*z*/){}
   void         TGQt::glVertex3fv(const Float_t  * /*vert*/){}
   void         TGQt::glIndexi(Int_t  /*index*/){}
   void         TGQt::glPointSize(Float_t  /*size*/){}
   void         TGQt::glLineWidth(Float_t  /*width*/){}
   void         TGQt::glDeleteLists(UInt_t  /*list*/, Int_t  /*sizei*/){}
   UInt_t       TGQt::glGenLists(UInt_t  /*list*/){return UInt_t(-1);}
   void         TGQt::glCallList(UInt_t  /*list*/){}
   void         TGQt::glMatrixMode(UInt_t  /*mode*/){}
   void         TGQt::glLoadIdentity(){}
   void         TGQt::glFrustum(Double_t  /*min_0*/, Double_t  /*max_0*/, Double_t  /*min_1*/,
                                  Double_t  /*max_1*/, Double_t  /*dnear*/, Double_t  /*dfar*/){}
   void         TGQt::glOrtho(Double_t  /*min_0*/, Double_t  /*max_0*/, Double_t  /*min_1*/,
                                Double_t  /*max_1*/, Double_t  /*dnear*/, Double_t  /*dfar*/){}
   void         TGQt::glCullFace(UInt_t  /*mode*/){}
   void         TGQt::glPolygonMode(UInt_t  /*face*/, UInt_t  /*mode*/){}
   void         TGQt::glLoadMatrixd(const Double_t  * /*matrix*/){}
   void         TGQt::glShadeModel(UInt_t  /*mode*/){}
   void         TGQt::glNormal3fv(const Float_t  * /*norm*/){}
#endif
