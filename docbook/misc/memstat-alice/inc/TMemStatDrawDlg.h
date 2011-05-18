// @(#)root/memstat:$Name$:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 31/05/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMemStatDrawDlg
#define ROOT_TMemStatDrawDlg

// STD
#include <vector>
#include <string>
// ROOT
#ifndef ROOT_RQ_OBJECT
#include "RQ_OBJECT.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TMemStat;
class TGComboBox;
class TGNumberEntry;
class TRootEmbeddedCanvas;

typedef std::vector<std::string> StringVector_t;

class TMemStatDrawDlg
{
   RQ_OBJECT("TMemStatDrawDlg")

public:
   TMemStatDrawDlg(TGCompositeFrame *parent, TMemStat *MemStat);
   virtual ~TMemStatDrawDlg();

   // slots
   void HandleDrawMemStat();

private:
   void PlaceCtrls(TGCompositeFrame *frame);
   void PlaceLBoxCtrl(TGCompositeFrame *frame, TGComboBox **box ,
                      const std::string &Label, const StringVector_t &Vealues, Int_t resource);
   void PlaceDeepCtrl(TGCompositeFrame *frame);
   void PlaceEmbeddedCanvas(TGCompositeFrame *frame);
   void ReDraw();

private:
   TMemStat *fMemStat;
   TGComboBox *fboxOrder;
   TGComboBox *fboxSortStat;
   TGComboBox *fboxSortStamp;
   TGNumberEntry *fNmbStackDeep;
   TGNumberEntry *fNmbSortDeep;
   TGNumberEntry *fNmbMaxLength;
   TRootEmbeddedCanvas *fEc;
};

#endif
