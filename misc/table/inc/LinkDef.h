/* @(#)root/table:$Id$ */

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

#pragma link C++ class TDataSet+;
#pragma link C++ class TDataSetIter;
#pragma link C++ class TFileSet+;
#pragma link C++ class TFileIter;
#pragma link C++ class TObjectSet+;
#pragma link C++ class TCL;

#pragma link C++ class TTableIter;
#pragma link C++ class TTable-;
#pragma link C++ class TTable::iterator!-;
#pragma link C++ class TGenericTable-;
#pragma link C++ class TGenericTable::iterator!-;

#pragma link C++ class TIndexTable-;
#pragma link C++ class TIndexTable::iterator!-;

#pragma link C++ class TChair+;
#pragma link C++ class TTableSorter;
#pragma link C++ class TTableDescriptor-;
#pragma link C++ class tableDescriptor_st+;
#pragma link C++ class TColumnView;
#pragma link C++ class TTableMap-;
#pragma link C++ class TTablePadView3D;


#pragma link C++ class TResponseTable-;


#pragma link C++ class TPoints3D+;
#pragma link C++ class TPolyLineShape+;
#pragma link C++ class TVolume+;
#pragma link C++ class TVolumePosition-;
#pragma link C++ class TVolumeView+;
#pragma link C++ class TVolumeViewIter;
#pragma link C++ class TPointsArray3D-;
#pragma link C++ class TTablePoints+;
#pragma link C++ class TTable3Points+;

#pragma link C++ function operator<<(ostream &, const TVolumePosition &);
#endif
