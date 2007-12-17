// @(#):$Id$
// Author: M.Gheata 
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoPconEditor
#define ROOT_TGeoPconEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoPconEditor                                                      //
//                                                                      //
//  Editor for a TGeoPcon.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGeoGedFrame
#include "TGeoGedFrame.h"
#endif

class TGeoPcon;
class TGeoPconSection;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class TGCanvas;
class TString;

class TGeoPconEditor : public TGeoGedFrame {

protected:
   Int_t                fNsecti;            // Initial number of sections
   Double_t             fPhi1i;             // Initial Phi1
   Double_t             fDPhii;             // Initial Dphi
   Double_t            *fZi;                // Initial Z positions
   Double_t            *fRmini;             // Initial Rmin values
   Double_t            *fRmaxi;             // Initial Rmax values   
   Int_t                fNsections;         // Number of Z sections
   TObjArray           *fSections;          // List of section frames
   TGCanvas            *fCan;               // sections container
   TGeoPcon            *fShape;             // Shape object
   Bool_t               fIsModified;        // Flag that volume was modified
   Bool_t               fIsShapeEditable;   // Flag that the shape can be changed

   TGLayoutHints       *fLHsect;            // Layout hints for sections
   TGTextEntry         *fShapeName;         // Shape name text entry
   TGNumberEntry       *fENz;               // Number entry for nsections
   TGNumberEntry       *fEPhi1;             // Number entry for phi1
   TGNumberEntry       *fEDPhi;             // Number entry for dphi  
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fUndo;              // Undo-Button
   TGCompositeFrame    *fBFrame;            // Frame containing Apply/Undo
   TGCheckButton       *fDelayed;           // Check button for delayed draw
   TGCompositeFrame    *fDFrame;            // Frame containing Delayed draw

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots
   Bool_t       CheckSections(Bool_t change=kFALSE);
   Bool_t       IsDelayed() const;   
   void         CreateSections(Int_t inew);
   void         UpdateSections();
   virtual void CreateEdges() {;}
public:
   TGeoPconEditor(const TGWindow *p = 0,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoPconEditor();
   virtual void   SetModel(TObject *obj);
   
   void           DoModified();
   void           DoName();
   void           DoNz();
   void           DoPhi();
   void           DoSectionChange(Int_t i);
   virtual void   DoApply();
   virtual void   DoUndo();
   
   ClassDef(TGeoPconEditor,0)   // TGeoPcon editor
};   
  
//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoPconSection                                                     //
//                                                                      //
//  Utility frame used by TGeoPcon editor.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGeoPconSection : public TGCompositeFrame, public TGWidget  {

protected:
   Int_t                fNumber;            // Id for the section
   TGNumberEntry       *fEZ;                // Number entry for Z position
   TGNumberEntry       *fERmin;             // Number entry for Rmin position
   TGNumberEntry       *fERmax;             // Number entry for Rmax position

   virtual void    ConnectSignals2Slots();
   
public:
   TGeoPconSection(const TGWindow *p, UInt_t w, UInt_t h, Int_t id);
   virtual ~TGeoPconSection();
   void         HideDaughters();
   Double_t     GetZ() const;
   Double_t     GetRmin() const;
   Double_t     GetRmax() const;
   void         SetZ(Double_t z);
   void         SetRmin(Double_t rmin);
   void         SetRmax(Double_t rmax);

   void           DoZ();
   void           DoRmin();
   void           DoRmax();

   virtual void Changed(Int_t i);   // *SIGNAL*

   ClassDef(TGeoPconSection,0)   // TGeoPcon section
};      
#endif                    
