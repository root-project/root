/* @(#)root/base:$Id$ */

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
#pragma link C++ global kMaxUChar;
#pragma link C++ global kMaxChar;
#pragma link C++ global kMinChar;
#pragma link C++ global kMaxUShort;
#pragma link C++ global kMaxShort;
#pragma link C++ global kMinShort;
#pragma link C++ global kMaxUInt;
#pragma link C++ global kMaxInt;
#pragma link C++ global kMinInt;
#pragma link C++ global kMaxULong;
#pragma link C++ global kMaxLong;
#pragma link C++ global kMinLong;
#pragma link C++ global kMaxULong64;
#pragma link C++ global kMaxLong64;
#pragma link C++ global kMinLong64;
#pragma link C++ global kBitsPerByte;
#pragma link C++ global kNPOS;
#pragma link C++ global kInfo;
#pragma link C++ global kWarning;
#pragma link C++ global kError;
#pragma link C++ global kSysError;
#pragma link C++ global kFatal;

#pragma link C++ enum EObjBits;
#pragma link C++ enum EEnvLevel;
#pragma link C++ enum EColor;
#pragma link C++ enum ELineStyle;
#pragma link C++ enum EMarkerStyle;

#pragma link C++ global gROOT;
#pragma link C++ global gEnv;
#pragma link C++ global gSystem;
#pragma link C++ global gPluginMgr;
#pragma link C++ global gApplication;
#pragma link C++ global gBenchmark;
#pragma link C++ global gDirectory;
#pragma link C++ global gDebug;
#pragma link C++ global gErrorIgnoreLevel;
#pragma link C++ global gErrorAbortLevel;
#pragma link C++ global gPrintViaErrorHandler;
#pragma link C++ global gStyle;
#pragma link C++ global gVirtualPS;
#pragma link C++ global gRootDir;
#pragma link C++ global gProgName;
#pragma link C++ global gProgPath;

#pragma link C++ function Info;
#pragma link C++ function Warning;
#pragma link C++ function Error;
#pragma link C++ function SysError;
#pragma link C++ function Fatal;
#pragma link C++ function Obsolete;
#pragma link C++ function Form;
#pragma link C++ function Printf;
#pragma link C++ function Strip;
#pragma link C++ function StrDup;
#pragma link C++ function Compress;
#pragma link C++ function EscChar;
#pragma link C++ function UnEscChar;
#pragma link C++ function Hash(const char*);
#pragma link C++ function Hash(const TString&);
#pragma link C++ function Hash(const TString*);
#pragma link C++ function ToLower(const TString&);
#pragma link C++ function ToUpper(const TString&);
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
#pragma link C++ function operator!=(const TString&,const TString&);
#pragma link C++ function operator!=(const TString&,const char*);
#pragma link C++ function operator!=(const char*,const TString&);
#pragma link C++ function operator>>(istream&,TString&);
#pragma link C++ function operator<<(ostream&,const TString&);
//#pragma link C++ function operator>>(TBuffer&,TString&);
//#pragma link C++ function operator<<(TBuffer&,const TString&);
//#pragma link C++ function operator>>(TBuffer&,const TObject*&);
//#pragma link C++ function operator<<(TBuffer&,const TObject*);

#pragma link C++ function operator==(const TDatime&,const TDatime&);
#pragma link C++ function operator!=(const TDatime&,const TDatime&);
#pragma link C++ function operator<(const TDatime&,const TDatime&);
#pragma link C++ function operator<=(const TDatime&,const TDatime&);
#pragma link C++ function operator>(const TDatime&,const TDatime&);
#pragma link C++ function operator>=(const TDatime&,const TDatime&);

#pragma link C++ nestedtypedef;
#pragma link C++ namespace ROOT;
#pragma create TClass TMath;
#pragma link C++ global ROOT_TMathBase;
#pragma link C++ typedef ShowMembersFunc_t;
#pragma link C++ typedef ROOT::NewFunc_t;
#pragma link C++ typedef ROOT::NewArrFunc_t;
#pragma link C++ typedef ROOT::DelFunc_t;
#pragma link C++ typedef ROOT::DelArrFunc_t;
#pragma link C++ typedef ROOT::DesFunc_t;
#pragma link C++ typedef Float16_t;
#pragma link C++ typedef Double32_t;

#pragma link C++ class TApplication;
#pragma link C++ class TApplicationImp;
#pragma link C++ class TAttFill+;
#pragma link C++ class TAttLine+;
#pragma link C++ class TAttMarker+;
#pragma link C++ class TAttPad-;
#pragma link C++ class TAttAxis-;
#pragma link C++ class TAttText+;
#pragma link C++ class TAtt3D+;
#pragma link C++ class TAttBBox+;
#pragma link C++ class TBenchmark+;
#pragma link C++ class TBrowser+;
#pragma link C++ class TBrowserImp+;
#pragma link C++ class TBuffer;
#pragma link C++ class TRootIOCtor+;
#pragma link C++ class TCanvasImp;
#pragma link C++ class TColor+;
#pragma link C++ class TContextMenu+;
#pragma link C++ class TContextMenuImp+;
#pragma link C++ class TControlBarImp+;
#pragma link C++ class TInspectorImp+;
#pragma link C++ class TDatime-;
#pragma link C++ class TDirectory;
#pragma link C++ class TEnv+;
#pragma link C++ class TEnvRec+;
#pragma link C++ class TFileHandler+;
#pragma link C++ class TGuiFactory;
#pragma link C++ class TStyle+;
#pragma link C++ class TVirtualX+;
#pragma link C++ class TVirtualPad-;
// Those are NOT going to be saved ... so no need for a +
#pragma link C++ class TVirtualViewer3D;
#pragma link C++ class TBuffer3D;
#pragma link C++ class TGLManager;
#pragma link C++ class TVirtualGLPainter;
#pragma link C++ class TVirtualGLManip;
#pragma link C++ class TVirtualPS;
#pragma link C++ class TGLPaintDevice;
#pragma link C++ class TVirtualPadPainter;

#pragma link C++ class TVirtualPadEditor;

#pragma link C++ class TVirtualFFT;

#endif
