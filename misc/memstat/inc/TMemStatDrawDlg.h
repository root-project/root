// @(#)root/memstat:$Name$:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 31/05/2008
#ifndef _ROOT_TMEMSTATDRAWDLG_H
#define _ROOT_TMEMSTATDRAWDLG_H

// STD
#include <vector>
#include <string>
// ROOT
#include "RQ_OBJECT.h"
#include "TGFrame.h"

class TMemStat;
class TGComboBox;
class TGNumberEntry;
class TRootEmbeddedCanvas;

typedef std::vector<std::string> StringVector_t;

class TMemStatDrawDlg
{
        RQ_OBJECT("TMemStatDrawDlg")

    public:
        TMemStatDrawDlg(const TGWindow *p, const TGWindow *main, TMemStat *MemStat);
        virtual ~TMemStatDrawDlg();

        // slots
        void HandleDrawMemStat();
        void CloseWindow();

    private:
        void PlaceCtrls(TGCompositeFrame *frame);
        void PlaceLBoxCtrl(TGCompositeFrame *frame, TGComboBox **box ,
                           const std::string &Label, const StringVector_t &Vealues, Int_t resource);
        void PlaceDeepCtrl(TGCompositeFrame *frame);
        void PlaceEmbeddedCanvas(TGCompositeFrame *frame);
        void ReDraw();

    private:
        TGTransientFrame *fMain;
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
