/* @(#)root/graf:$Name:  $:$Id: LinkDef2.h,v 1.11 2005/07/05 12:36:06 brun Exp $ */

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

#pragma link C++ class TGraphSmooth+;
#pragma link C++ class TLatex+;
#pragma link C++ class TLegend+;
#pragma link C++ class TLegendEntry+;
#pragma link C++ class TLink+;
#pragma link C++ class TPoints+;
#pragma link C++ class TSpline-;
#pragma link C++ class TSpline5-;
#pragma link C++ class TSpline3-;
#pragma link C++ class TSplinePoly+;
#pragma link C++ class TSplinePoly5+;
#pragma link C++ class TSplinePoly3+;
#pragma link C++ class TImage;
#pragma link C++ class TAttImage;
#pragma link C++ class TImagePlugin;
#pragma link C++ class TImagePalette;
#pragma link C++ class TPaletteEditor;
#pragma link C++ class TText-;
#pragma link C++ class TTF;
#pragma link C++ class TGraphPolar;
#pragma link C++ class TGraphPolargram;

#pragma link C++ global gHistImagePalette;
#pragma link C++ global gWebImagePalette;

#pragma link C++ enum TImage::EImageFileTypes;
#pragma link C++ enum TImage::EText3DType;
#pragma link C++ enum TImage::ECharType;
#pragma link C++ enum TImage::ETileType;
#pragma link C++ enum TImage::ECoordMode;
#pragma link C++ enum TImage::EColorChan;

#pragma link C++ function operator+(const TImage&,const TImage&);
#pragma link C++ function operator/(const TImage&,const TImage&);

#endif
