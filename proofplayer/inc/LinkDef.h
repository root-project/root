/* @(#)root/proof:$Name:  $:$Id: LinkDef.h,v 1.2 2007/03/17 18:04:02 rdm Exp $ */

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

#pragma link C++ class TProofPlayer+;
#pragma link C++ class TProofPlayerLocal+;
#pragma link C++ class TProofPlayerRemote+;
#pragma link C++ class TProofPlayerSlave+;
#pragma link C++ class TProofPlayerSuperMaster+;

#pragma link C++ class TVirtualPacketizer+;
#pragma link C++ class TPacketizer+;
#pragma link C++ class TPacketizerDev+;
#pragma link C++ class TPacketizerProgressive+;
#pragma link C++ class TAdaptivePacketizer+;

#pragma link C++ class TPerfStats;
#pragma link C++ class TPerfEvent+;

#pragma link C++ class TProofDraw+;
#pragma link C++ class TProofDrawEventList+;
#pragma link C++ class TProofDrawHist+;
#pragma link C++ class TProofDrawProfile+;
#pragma link C++ class TProofDrawProfile2D+;
#pragma link C++ class TProofDrawGraph+;
#pragma link C++ class TProofDrawPolyMarker3D+;
#pragma link C++ class TProofDrawListOfGraphs+;
#pragma link C++ class TProofDrawListOfPolyMarkers3D+;
#pragma link C++ class TProofDrawListOfGraphs::Point3D_t+;
#pragma link C++ class TProofDrawListOfPolyMarkers3D::Point4D_t+;
#pragma link C++ class std::vector<TProofDrawListOfGraphs::Point3D_t>+;
#pragma link C++ class std::vector<TProofDrawListOfPolyMarkers3D::Point4D_t>+;
#pragma link C++ class TProofVectorContainer<TProofDrawListOfGraphs::Point3D_t>+;
#pragma link C++ class TProofVectorContainer<TProofDrawListOfPolyMarkers3D::Point4D_t>+;

#pragma link C++ class TProofLimitsFinder;
#pragma link C++ class TDrawFeedback+;
#pragma link C++ class TFileMerger+;

#endif
