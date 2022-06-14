/*
 * $Header$
 * $Log$
 */

#ifndef __XSREACTION_DLG_H
#define __XSREACTION_DLG_H

#include <TGTab.h>
#include <TGFrame.h>
#include <TGButton.h>
#include <TGLayout.h>
#include <TGListBox.h>
#include <TGComboBox.h>
#include <TGTextView.h>
#include <TGTextEntry.h>

#include "NdbMTReactionXS.h"

#include "XSStepButton.h"
#include "XSElementList.h"
#include "XSPeriodicTable.h"

/* =========== XSReactionDlg ============== */
class XSReactionDlg : public TGTransientFrame
{
protected:

   UInt_t      Z;

   const TGWindow   *mainWindow;

   TGHorizontalFrame   *frm1,
            *frm2,
            *frm3,
            *frm4,
            *frm5;

   TGCompositeFrame   *Vfrm1,
            *Vfrm2,
            *Vfrm3;

   TGLayoutHints   *lHFixed,
         *lHExpX,
         *lHExpY,
         *lHExpXY,
         *lHBot,
         *lHExpXCen,
         *lHFixedCen;

   // ---- Material Items ----
   TGGroupFrame   *materialGroup;

   TGLabel      *elementLbl;
   TGTextBuffer   *elementBuf; //!
   TGTextEntry   *elementText;
   XSStepButton   *elementStep;

   TGLabel      *nameLbl,
         *mnemonicLbl,

         *chargeLbl,
         *zLbl,

         *massLbl,
         *massValLbl,

         *isotopeLbl,

         *densityLbl,
         *densityValLbl,

         *meltingPtLbl,
         *meltingValLbl,
         *boilingPtLbl,
         *boilingValLbl,

         *oxidationLbl,
         *oxidationValLbl,

         *isotopeInfoLbl,
         *isotopeInfoValLbl;

   TGButton   *ptableButton;

   TGComboBox   *isotopeCombo;

   // ----- Reaction ----
   TGGroupFrame   *reactionGroup;

   TGLabel      *projectileLbl,
         *temperatureLbl,
         *databaseLbl,
         *reactionLbl,
         *reactionInfoLbl,
         *reactionInfoValLbl;

   TGComboBox   *projectileCombo,
         *temperatureCombo,
         *databaseCombo;

   TGListBox   *reactionList;

   // ----- Options -----
   TGGroupFrame   *optionGroup;

   TGLabel      *lineWidthLbl,
         *lineColorLbl,
         *markerStyleLbl,
         *markerColorLbl,
         *markerSizeLbl,
         *errorbarColorLbl;

   TGComboBox   *lineWidthCombo,
         *lineColorCombo,
         *markerStyleCombo,
         *markerColorCombo,
         *markerSizeCombo,
         *errorbarColorCombo;

   // ----- Info Group -----
   TGGroupFrame   *infoGroup;

   TGTextView   *infoView;


   // ----- Execution Buttons ----
   TGHorizontalFrame   *buttonFrame;
   TGButton      *okButton,
            *execButton,
            *resetButton,
            *closeButton;


public:
   XSReactionDlg(const TGWindow *p,
           const TGWindow *main, UInt_t initZ, UInt_t w, UInt_t h);
   ~XSReactionDlg();

protected:
      void   InitColorCombo(TGComboBox *cb);
      void   InitCombos();
   const   char*   GetString(int box);
      char*   CreatePath(int option);
      int   UpdateContainer( TGListBox *lb, char *path, int option);
      void   UpdateCurIsotope();
      void   UpdateIsotopes();
      void   UpdateProjectile();
      void   UpdateDatabase();
      void   UpdateReactions();

      void   SetElement(UInt_t aZ);
      void   ElementEntryChanged();

      void   UpdateGraph(NdbMTReactionXS *xs);
      Bool_t   ExecCommand();

   virtual void   CloseWindow();
      Bool_t   ProcessButton(Longptr_t param1, Longptr_t param2);
      Bool_t   ProcessCombo(Longptr_t param1, Longptr_t param2);
   virtual Bool_t   ProcessMessage(Longptr_t msg, Longptr_t param1, Longptr_t param2);

   //ClassDef(XSReactionDlg,1)
}; // XSReactionDlg

#endif
