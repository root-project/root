/* @(#)root/proof:$Name:  $:$Id: LinkDef.h,v 1.36 2006/12/03 23:34:03 rdm Exp $ */

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

#pragma link C++ nestedtypedefs;
#pragma link C++ nestedclasses;

#pragma link C++ global gProofServ;

#pragma link C++ global gProofDebugMask;
#pragma link C++ global gProofDebugLevel;

#pragma link C++ class TDSet+;
#pragma link C++ class TDSetElement+;
#pragma link C++ class TProofChain+;
#pragma link C++ class TProofMgr;
#pragma link C++ class TProofDesc;
#pragma link C++ class TProof;
#pragma link C++ class TProofCondor;
#pragma link C++ class TProofSuperMaster;
#pragma link C++ class TSlaveInfo+;
#pragma link C++ class TProofServ;
#pragma link C++ class TProofDebug;
#pragma link C++ class TProofLimitsFinder;
#pragma link C++ class TProofLog;
#pragma link C++ class TProofLogElem;
#pragma link C++ class TSlave;
#pragma link C++ class TProofPlayer+;
#pragma link C++ class TProofPlayerLocal+;
#pragma link C++ class TProofPlayerRemote+;
#pragma link C++ class TProofPlayerSlave+;
#pragma link C++ class TProofPlayerSuperMaster+;
#pragma link C++ class TProofQueryResult+;
#pragma link C++ class TEventIter+;
#pragma link C++ class TEventIterObj+;
#pragma link C++ class TEventIterTree+;
#pragma link C++ class TDSetProxy+;
#pragma link C++ class TVirtualPacketizer+;
#pragma link C++ class TPacketizer+;
#pragma link C++ class TPacketizerDev+;
#pragma link C++ class TPacketizerProgressive+;
#pragma link C++ class TAdaptivePacketizer+;
#pragma link C++ class TCondor+;
#pragma link C++ class TCondorSlave+;
#pragma link C++ class TProofDraw+;
#pragma link C++ class TProofDrawEventList+;
#pragma link C++ class TProofDrawHist+;
#pragma link C++ class TProofDrawProfile+;
#pragma link C++ class TProofDrawProfile2D+;
#pragma link C++ class TProofDrawGraph+;
#pragma link C++ class TProofDrawPolyMarker3D+;
#pragma link C++ class TProofDrawListOfGraphs+;
#pragma link C++ class TProofDrawListOfPolyMarkers3D+;
#pragma link C++ class TProofNodeInfo;
#pragma link C++ class TProofResources;
#pragma link C++ class TProofResourcesStatic;
#pragma link C++ class TDrawFeedback+;
#pragma link C++ class TPerfStats;
#pragma link C++ class TPerfEvent+;
#pragma link C++ class TStatus+;
#pragma link C++ class TFileMerger+;
#pragma link C++ class TProofDrawListOfGraphs::Point3D_t+;
#pragma link C++ class TProofDrawListOfPolyMarkers3D::Point4D_t+;
#pragma link C++ class std::vector<TProofDrawListOfGraphs::Point3D_t>+;
#pragma link C++ class std::vector<TProofDrawListOfPolyMarkers3D::Point4D_t>+;
#pragma link C++ class TProofVectorContainer<TProofDrawListOfGraphs::Point3D_t>+;
#pragma link C++ class TProofVectorContainer<TProofDrawListOfPolyMarkers3D::Point4D_t>+;
#pragma link C++ class std::pair<TDSet*, TString>;
#pragma link C++ class std::pair<TDSetElement*, TString>;

#pragma link C++ global gProof;

#endif
