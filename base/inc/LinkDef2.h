/* @(#)root/base:$Name:  $:$Id: LinkDef2.h,v 1.5 2000/11/13 19:46:13 rdm Exp $ */

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

#pragma link C++ global gTQSender;
#pragma link C++ global gTQSlotParams;

#pragma link C++ enum EAccessMode;

#pragma link C++ function operator+(const TTime&,const TTime&);
#pragma link C++ function operator-(const TTime&,const TTime&);
#pragma link C++ function operator*(const TTime&,const TTime&);
#pragma link C++ function operator/(const TTime&,const TTime&);

#pragma link C++ function operator==(const TTime&,const TTime&);
#pragma link C++ function operator!=(const TTime&,const TTime&);
#pragma link C++ function operator<(const TTime&,const TTime&);
#pragma link C++ function operator<=(const TTime&,const TTime&);
#pragma link C++ function operator>(const TTime&,const TTime&);
#pragma link C++ function operator>=(const TTime&,const TTime&);

#pragma link C++ function ConnectCINT(TQObject*,char*,char*);

#pragma link C++ class TExec+;
#pragma link C++ class TFolder+;
#pragma link C++ class TFree;
#pragma link C++ class TKey-;
#pragma link C++ class TKeyMapFile;
#pragma link C++ class TMapFile;
#pragma link C++ class TMapRec;
#pragma link C++ class TMath;
#pragma link C++ class TMemberInspector;
#pragma link C++ class TMessageHandler+;
#pragma link C++ class TNamed+;
#pragma link C++ class TObjString+;
#pragma link C++ class TObject-;
#pragma link C++ class TProcessEventTimer;
#pragma link C++ class TRandom+;
#pragma link C++ class TRandom2+;
#pragma link C++ class TRandom3-;
#pragma link C++ class TROOT;
#pragma link C++ class TRealData+;
#pragma link C++ class TRegexp;
#pragma link C++ class TSignalHandler;
#pragma link C++ class TStopwatch+;
#pragma link C++ class TStorage;
#pragma link C++ class TString-;
#pragma link C++ class TStringLong-;
#pragma link C++ class TSubString;
#pragma link C++ class TSysEvtHandler;
#pragma link C++ class TSystem+;
#pragma link C++ class TSystemFile+;
#pragma link C++ class TSystemDirectory+;
#pragma link C++ class TTask+;
#pragma link C++ class TTime;
#pragma link C++ class TTimer;
#pragma link C++ class TQObject-;
#pragma link C++ class TQObjSender;
#pragma link C++ class TQClass;
#pragma link C++ class TQConnection;

#endif
