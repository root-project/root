/* @(#)root/base:$Name:  $:$Id: LinkDef1.h,v 1.2 2000/08/18 11:00:59 brun Exp $ */

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

#pragma link C++ global kTRUE;
#pragma link C++ global kFALSE;
#pragma link C++ global kNPOS;
#pragma link C++ global kWarning;
#pragma link C++ global kError;
#pragma link C++ global kSysError;
#pragma link C++ global kFatal;

#pragma link C++ enum EObjBits;
#pragma link C++ enum EEnvLevel;

#pragma link C++ global gROOT;
#pragma link C++ global gEnv;
#pragma link C++ global gSystem;
#pragma link C++ global gBenchmark;
#pragma link C++ global gDirectory;
#pragma link C++ global gFile;
#pragma link C++ global gRandom;
#pragma link C++ global gDebug;
#pragma link C++ global gErrorIgnoreLevel;
#pragma link C++ global gStyle;
#pragma link C++ global gVirtualGL;
#pragma link C++ global gVirtualX;
#pragma link C++ global gVirtualPS;

#pragma link C++ function Strip;
#pragma link C++ function StrDup;
#pragma link C++ function Compress;
#pragma link C++ function EscChar;
#pragma link C++ function UnEscChar;
#pragma link C++ function operator+(const TString&,const TString&);
#pragma link C++ function operator+(const TString&,const char*);
#pragma link C++ function operator+(const char*,const TString&);
#pragma link C++ function operator+(const TString&,char);
#pragma link C++ function operator+(const TString&,Long_t);
#pragma link C++ function operator+(const TString&,ULong_t);
#pragma link C++ function operator+(char,const TString&);
#pragma link C++ function operator+(Long_t,const TString&);
#pragma link C++ function operator+(ULong_t,const TString&);
#pragma link C++ function operator==(const TString&,const TString&);
#pragma link C++ function operator==(const TString&,const char*);
#pragma link C++ function operator==(const char*,const TString&);
#pragma link C++ function operator!=(const TString&,const TString&)
#pragma link C++ function operator!=(const TString&,const char*)
#pragma link C++ function operator!=(const char*,const TString&);
#pragma link C++ function operator>>(istream&,TString&);
#pragma link C++ function operator<<(ostream&,const TString&);
//#pragma link C++ function operator>>(TBuffer&,const TObject*&);
//#pragma link C++ function operator<<(TBuffer&,const TObject*);

#pragma link C++ function operator==(const TDatime&,const TDatime&);
#pragma link C++ function operator!=(const TDatime&,const TDatime&);
#pragma link C++ function operator<(const TDatime&,const TDatime&);
#pragma link C++ function operator<=(const TDatime&,const TDatime&);
#pragma link C++ function operator>(const TDatime&,const TDatime&);
#pragma link C++ function operator>=(const TDatime&,const TDatime&);

#pragma link C++ class TApplication;
#pragma link C++ class TApplicationImp;
#pragma link C++ class TAttFill;
#pragma link C++ class TAttLine;
#pragma link C++ class TAttMarker;
#pragma link C++ class TAttPad-;
#pragma link C++ class TAttAxis-;
#pragma link C++ class TAttText;
#pragma link C++ class TAtt3D;
#pragma link C++ class TBenchmark;
#pragma link C++ class TBrowser;
#pragma link C++ class TBrowserImp;
#pragma link C++ class TBuffer;
#pragma link C++ class TCanvasImp;
#pragma link C++ class TColor+;
#pragma link C++ class TContextMenu;
#pragma link C++ class TContextMenuImp;
#pragma link C++ class TControlBarImp;
#pragma link C++ class TInspectorImp;
#pragma link C++ class TDatime-;
#pragma link C++ class TDirectory-;
#pragma link C++ class TEnv;
#pragma link C++ class TFile-;
#pragma link C++ class TFileHandler;
#pragma link C++ class TGuiFactory;
#pragma link C++ class TPadView3D;
#pragma link C++ class TStyle+;
#pragma link C++ class TView-;
#pragma link C++ class TVirtualX;
#pragma link C++ class TVirtualFitter;
#pragma link C++ class TVirtualPad;
#pragma link C++ class TVirtualGL;
#pragma link C++ class TVirtualPS;

#endif
