/* @(#)root/base:$Name:  $:$Id: LinkDef1.h,v 1.46 2007/02/04 17:39:44 brun Exp $ */

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

//was in base/inc/LinkDef1.h
#pragma link C++ global gFile;

//was in base/inc/LinkDef1.h
#pragma link C++ class TBufferFile;
#pragma link C++ class TDirectoryFile-;
#pragma link C++ class TFile-;
#pragma link C++ class TFileCacheRead+;
#pragma link C++ class TFileCacheWrite+;

//was in base/inc/LinkDef2.h
#pragma link C++ class TKey-;
#pragma link C++ class TMapFile;

//was in base/inc/LinkDef3.h
#pragma link C++ class TArchiveFile+;
#pragma link C++ class TArchiveMember+;
#pragma link C++ class TZIPFile+;
#pragma link C++ class TZIPMember+;

//was in meta/inc/LinkDef.h
#pragma link C++ class TStreamerInfo-;

//was in cont/inc/LinkDef.h
#pragma link C++ class TCollectionProxyFactory-;
#pragma link C++ class TEmulatedCollectionProxy-;
#pragma link C++ class TEmulatedMapProxy-;
#pragma link C++ class TGenCollectionProxy-;
#pragma link C++ class TGenCollectionProxy::Value-;
#pragma link C++ class TGenCollectionProxy::Method-;
#pragma link C++ class TCollectionStreamer-;
#pragma link C++ class TCollectionClassStreamer-;
#pragma link C++ class TCollectionMemberStreamer-;

#endif
