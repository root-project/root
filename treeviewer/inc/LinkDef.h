/* @(#)root/treeviewer:$Name:  $:$Id: LinkDef.h,v 1.9 2005/11/21 00:31:40 rdm Exp $ */

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

#pragma link C++ global gSessionViewer;

#pragma link C++ class TTreeViewer+;
#pragma link C++ class TTVLVContainer+;
#pragma link C++ class TTVLVEntry+;
#pragma link C++ class TGSelectBox+;
#pragma link C++ class TGItemContext+;
#pragma link C++ class TTVRecord+;
#pragma link C++ class TTVSession+;
#pragma link C++ class TSessionLogView+;
#pragma link C++ class TSessionServerFrame+;
#pragma link C++ class TSessionFrame+;
#pragma link C++ class TEditQueryFrame+;
#pragma link C++ class TSessionQueryFrame+;
#pragma link C++ class TSessionOutputFrame+;
#pragma link C++ class TSessionInputFrame+;
#pragma link C++ class TSessionViewer+;
#pragma link C++ class TQueryDescription+;
#pragma link C++ class TSessionDescription+;
#pragma link C++ class TPackageDescription+;
#pragma link C++ class TNewQueryDlg+;
#pragma link C++ class TNewChainDlg+;
#pragma link C++ class TUploadDataSetDlg+;

#endif
