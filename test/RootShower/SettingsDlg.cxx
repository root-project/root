// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

#include <stdlib.h>
#include <TGLabel.h>
#include <TRootHelpDialog.h>
#include <TGMsgBox.h>

#include "SettingsDlg.h"
#include "RootShower.h"
#include "constants.h"
#include "RSHelpText.h"

// definition of structure used to retrieve
// PDG number and particle name into the list
// of available primaries

typedef struct {
    Int_t       pdg_code;
    const char *pdg_name;
} str_choice_def;

str_choice_def choice_def[] = {
      {    22,  "gamma"       },
      {    11,  "e-"          },
      {   -11,  "e+"          },
      {    13,  "mu-"         },
      {   -13,  "mu+"         },
      {    15,  "tau-"        },
      {   -15,  "tau+"        },
      {   111,  "pi0"         },
      {   211,  "pi+"         },
      {  -211,  "pi-"         },
      {   221,  "Eta"         },
      {   321,  "K+"          },
      {  -321,  "K-"          },
      {   130,  "K(L)0"       },
      {   310,  "K(S)0"       },
      {   113,  "rho(770)0"   },
      {   213,  "rho(770)+"   },
      {  -213,  "rho(770)-"   },
      {   223,  "omega(782)0" },
      {   333,  "phi(1020)0"  },
      {   443,  "J/psi(1S)0"  },
      {   511,  "B0"          },
      {   513,  "B*0"         },
      {   521,  "B+"          },
      {  -521,  "B-"          },
      {   523,  "B*+"         },
      {  -523,  "B*-"         },
      {   531,  "B(s)0"       },
      {   411,  "D+"          },
      {  -411,  "D-"          },
      {   421,  "D0"          },
      {   431,  "D(s)+"       },
      {  -431,  "D(s)-"       },
      {   433,  "D(s)*+"      },
      {  -433,  "D(s)*-"      },
      {    24,  "W+"          },
      {   -24,  "W-"          },
      {    23,  "Z0"          },
      {     0,    0           }
};

//_________________________________________________
// SettingsDialog
//
// SettingsDialog is a dialog accessing the RootShowerhower parameters.

enum RootShowerSettingsTypes {
    Id1,
    Id2,
    Id3,
    Id4,
    Id5,
};


//______________________________________________________________________________
SettingsDialog::SettingsDialog(const TGWindow *p, const TGWindow *main, UInt_t w,
                       UInt_t h, UInt_t options)
    : TGTransientFrame(p, main, w, h, options)
{
    // Create a dialog window. A dialog window pops up with respect to its
    // "main" window.
    Int_t i;
    Char_t tmp[20];
    UInt_t wh1 = (UInt_t)(0.6 * h);
    UInt_t wh2 = h - wh1;

    fFrame1  = new TGHorizontalFrame(this, w, wh2, 0);

    fOkButton = new TGTextButton(fFrame1, "     &Ok     ", 1);
    fOkButton->Associate(this);
    fCancelButton = new TGTextButton(fFrame1, "   &Cancel   ", 2);
    fCancelButton->Associate(this);
    fHelpButton = new TGTextButton(fFrame1, "    &Help    ", 3);
    fHelpButton->Associate(this);

    fL1 = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX,
                           2, 2, 2, 2);
    fL2 = new TGLayoutHints(kLHintsBottom | kLHintsRight | kLHintsExpandX, 2, 2, 5, 1);

    fFrame1->AddFrame(fOkButton, fL1);
    fFrame1->AddFrame(fHelpButton, fL1);
    fFrame1->AddFrame(fCancelButton, fL1);

    fFrame1->Resize(150, fOkButton->GetDefaultHeight());
    AddFrame(fFrame1, fL2);

    //--------- create Tab widget and some composite frames for Tab testing
    fTab = new TGTab(this, 300, 300);
    fL3 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 10, 10, 10, 10);

    TGCompositeFrame *tf = fTab->AddTab("Target properties");
    tf->SetLayoutManager(new TGHorizontalLayout(tf));

    fF1 = new TGGroupFrame(tf, "Material", kVerticalFrame);
    tf->AddFrame(fF1, fL3);
    fF1->SetLayoutManager(new TGMatrixLayout(fF1, 0, 1, 10));

    fF1->AddFrame(fRad1 = new TGRadioButton(fF1, "Polystyrene", 81));
    fF1->AddFrame(fRad2 = new TGRadioButton(fF1, "Bismuth germanate", 82));
    fF1->AddFrame(fRad3 = new TGRadioButton(fF1, "Cesium iodide", 83));
    fF1->AddFrame(fRad4 = new TGRadioButton(fF1, "Sodium iodide ", 84));
    fRad1->SetToolTipText("Polystyrene ");
    fRad2->SetToolTipText("Bismuth germanate (BGO)");
    fRad3->SetToolTipText("Cesium iodide (CsI)");
    fRad4->SetToolTipText("Sodium iodide (NaI)");

    fRad1->Associate(this);
    fRad2->Associate(this);
    fRad3->Associate(this);
    fRad4->Associate(this);
    switch(gRootShower->fMaterial) {
        case Polystyrene:
            fRad1->SetState(kButtonDown);
            fRad2->SetState(kButtonUp);
            fRad3->SetState(kButtonUp);
            fRad4->SetState(kButtonUp);
            break;
        case BGO:
            fRad1->SetState(kButtonUp);
            fRad2->SetState(kButtonDown);
            fRad3->SetState(kButtonUp);
            fRad4->SetState(kButtonUp);
            break;
        case CsI:
            fRad1->SetState(kButtonUp);
            fRad2->SetState(kButtonUp);
            fRad3->SetState(kButtonDown);
            fRad4->SetState(kButtonUp);
            break;
        case NaI:
            fRad1->SetState(kButtonUp);
            fRad2->SetState(kButtonUp);
            fRad3->SetState(kButtonUp);
            fRad4->SetState(kButtonDown);
            break;
        default:
            fRad1->SetState(kButtonUp);
            fRad2->SetState(kButtonUp);
            fRad3->SetState(kButtonDown);
            fRad4->SetState(kButtonUp);
            break;
    }

    fF1->Resize(fF1->GetDefaultSize());

    // another matrix with text and buttons
    fF2 = new TGGroupFrame(tf, "Dimensions", kVerticalFrame);
    fF2->SetTitlePos(TGGroupFrame::kRight); // right aligned
    tf->AddFrame(fF2, fL3);

    fF2->SetLayoutManager(new TGMatrixLayout(fF2, 0, 2, 10));

    fF2->AddFrame(new TGLabel(fF2, "X [cm]"));
    fF2->AddFrame(fTxt1 = new TGTextEntry(fF2, new TGTextBuffer(100), Id1));
    fF2->AddFrame(new TGLabel(fF2, "Y [cm]"));
    fF2->AddFrame(fTxt2 = new TGTextEntry(fF2, new TGTextBuffer(100), Id2));
    fF2->AddFrame(new TGLabel(fF2, "Z [cm]"));
    fF2->AddFrame(fTxt3 = new TGTextEntry(fF2, new TGTextBuffer(100), Id3));
    fTxt1->Associate(this);
    fTxt2->Associate(this);
    fTxt3->Associate(this);
    fTxt1->Resize(60, fTxt1->GetDefaultHeight());
    fTxt2->Resize(60, fTxt2->GetDefaultHeight());
    fTxt3->Resize(60, fTxt3->GetDefaultHeight());
    sprintf(tmp,"%1.3f",gRootShower->fDimX);
    fTxt1->SetText(tmp);
    sprintf(tmp,"%1.3f",gRootShower->fDimY);
    fTxt2->SetText(tmp);
    sprintf(tmp,"%1.3f",gRootShower->fDimZ);
    fTxt3->SetText(tmp);

    tf = fTab->AddTab("Physics settings");
    tf->SetLayoutManager(new TGHorizontalLayout(tf));

    fF3 = new TGGroupFrame(tf, "Particle", kVerticalFrame);
    tf->AddFrame(fF3, fL3);
    fF3->SetLayoutManager(new TGMatrixLayout(fF3, 0, 1, 10));

    fF3->AddFrame(fListBox = new TGListBox(fF3, 89));

    for (i = 0; choice_def[i].pdg_name; i++) {
        fListBox->AddEntry(choice_def[i].pdg_name, i);
    }
    fFirstEntry = 0;
    fLastEntry  = 30;
    fListBox->Resize(120, 110);
    
    fF3->Resize(fF3->GetDefaultSize());

    fF4 = new TGGroupFrame(tf, "E0 / B", kVerticalFrame);
    fF4->SetTitlePos(TGGroupFrame::kRight); // right aligned
    tf->AddFrame(fF4, fL3);
    fF4->SetLayoutManager(new TGMatrixLayout(fF4, 0, 2, 10));

    fF4->AddFrame(new TGLabel(fF4, "E0 [GeV]"));
    fF4->AddFrame(fTxt4 = new TGTextEntry(fF4, new TGTextBuffer(100), Id4));
    fF4->AddFrame(new TGLabel(fF4, "B [kGauss]"));
    fF4->AddFrame(fTxt5 = new TGTextEntry(fF4, new TGTextBuffer(100), Id5));
    fTxt4->Associate(this);
    fTxt5->Associate(this);
    fTxt4->Resize(65, fTxt4->GetDefaultHeight());
    fTxt5->Resize(65, fTxt5->GetDefaultHeight());
    sprintf(tmp,"%1.4f",gRootShower->fE0);
    fTxt4->SetText(tmp);
    sprintf(tmp,"%1.4f",gRootShower->fB*3.308);
    fTxt5->SetText(tmp);
    
    TGLayoutHints *fL5 = new TGLayoutHints(kLHintsBottom | kLHintsExpandX |
                                          kLHintsExpandY, 2, 2, 5, 1);
    AddFrame(fTab, fL5);

    MapSubwindows();
    Resize(GetDefaultSize());

    fF2->Resize(fF2->GetDefaultWidth(),fF1->GetDefaultHeight());
    fF4->Resize(fF4->GetDefaultWidth(),fF3->GetDefaultHeight());

    for (i = 0; choice_def[i].pdg_name; i++) {
        if(gRootShower->fFirstParticle == choice_def[i].pdg_code) {
            fListBox->Select(i);
            fListBox->GetScrollBar()->SetPosition(i-2);
            break;
        }
    }
    fListBox->MapSubwindows();
    fListBox->Layout();
    // position relative to the parent's window
    Window_t wdum;
    Int_t ax, ay;
    gVirtualX->TranslateCoordinates(main->GetId(), GetParent()->GetId(),
                          (Int_t)(((TGFrame *) main)->GetWidth() - fWidth) >> 1,
                          (Int_t)(((TGFrame *) main)->GetHeight() - fHeight) >> 1,
                          ax, ay, wdum);
    Move(ax, ay);

    SetWindowName("Shower Settings");

    // make the message box non-resizable
    UInt_t width  = GetDefaultWidth();
    UInt_t height = GetDefaultHeight();
    SetWMSize(width, height);
    SetWMSizeHints(width, height, width, height, 0, 0);

    SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
                kMWMDecorMinimize | kMWMDecorMenu,
                kMWMFuncAll  | kMWMFuncResize    | kMWMFuncMaximize |
                kMWMFuncMinimize, kMWMInputModeless);
    MapWindow();
    fClient->WaitFor(this);
}

//______________________________________________________________________________
SettingsDialog::~SettingsDialog()
{
    // Delete test dialog widgets.

    delete fOkButton;
    delete fCancelButton;
    delete fHelpButton;
    delete fFrame1;
    delete fRad1; 
    delete fRad2;
    delete fRad3; 
    delete fRad4;
    delete fListBox;
    delete fF1; 
    delete fF2; 
    delete fF3; 
    delete fF4;
    delete fTxt1; 
    delete fTxt2; 
    delete fTxt3;
    delete fTxt4; 
    delete fTxt5;
    delete fTab;
    delete fL3;
    delete fL2; 
    delete fL1;
}


//______________________________________________________________________________
void SettingsDialog::CloseWindow()
{
    // Called when window is closed via the window manager.
    delete this;
}

//______________________________________________________________________________
Bool_t SettingsDialog::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
    // Process messages coming from widgets associated with the dialog.
    Int_t Selection;
    Int_t retval;
    TRootHelpDialog* hd;

    const Char_t *buf_tmp;

    switch (GET_MSG(msg)) {
      case kC_COMMAND:

         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               switch(parm1) {
                  case 1:
                     Selection = fListBox->GetSelected();
                     if(Selection > 37) {
                         new TGMsgBox(fClient->GetRoot(), this, "Particle selection",
                               "This particle is not implemented yet !",
                               kMBIconExclamation, kMBOk, &retval);
                         fListBox->Select(2);
                         break;
                     }
                     gRootShower->fFirstParticle = choice_def[Selection].pdg_code;
                     buf_tmp = fTxt1->GetBuffer()->GetString();
                     gRootShower->fDimX = atof(buf_tmp);
                     buf_tmp = fTxt2->GetBuffer()->GetString();
                     gRootShower->fDimY = atof(buf_tmp);
                     buf_tmp = fTxt3->GetBuffer()->GetString();
                     gRootShower->fDimZ = atof(buf_tmp);
                     buf_tmp = fTxt4->GetBuffer()->GetString();
                     gRootShower->fE0 = atof(buf_tmp);
                     buf_tmp = fTxt5->GetBuffer()->GetString();
                     gRootShower->fB = atof(buf_tmp);
                     gRootShower->fB /= 3.308;
                     if(fRad1->GetState() != kButtonUp) gRootShower->fMaterial = Polystyrene;
                     if(fRad2->GetState() != kButtonUp) gRootShower->fMaterial = BGO;
                     if(fRad3->GetState() != kButtonUp) gRootShower->fMaterial = CsI;
                     if(fRad4->GetState() != kButtonUp) gRootShower->fMaterial = NaI;
                     gRootShower->SettingsModified();
                  case 2:
                     CloseWindow();
                     break;
                  case 3:
                     hd = new TRootHelpDialog(this, "Help on Settings Dialog", 560, 400);
                     hd->SetText(gSettingsHelp);
                     hd->Popup();
                     fClient->WaitFor(hd);
                     break;
                  default:
                     break;
               }
               break;
            case kCM_RADIOBUTTON:
               switch (parm1) {
                  case 81:
                     fRad2->SetState(kButtonUp);
                     fRad3->SetState(kButtonUp);
                     fRad4->SetState(kButtonUp);
                     break;
                  case 82:
                     fRad1->SetState(kButtonUp);
                     fRad3->SetState(kButtonUp);
                     fRad4->SetState(kButtonUp);
                     break;
                  case 83:
                     fRad1->SetState(kButtonUp);
                     fRad2->SetState(kButtonUp);
                     fRad4->SetState(kButtonUp);
                     break;
                  case 84:
                     fRad1->SetState(kButtonUp);
                     fRad2->SetState(kButtonUp);
                     fRad3->SetState(kButtonUp);
                     break;
               }
               break;
            case kCM_TAB:
               break;
            default:
               break;
         }
         break;
      case kC_TEXTENTRY:
         switch (GET_SUBMSG(msg)) {
            case kTE_ENTER:
               switch (parm1) {
                  case Id1:
                     fTxt2->SetFocus();
                     break;
                  case Id2:
                     fTxt3->SetFocus();
                     break;
                  case Id3:
                     fTxt1->SetFocus();
                     break;
                  case Id4:
                     fTxt5->SetFocus();
                     break;
                  case Id5:
                     fTxt4->SetFocus();
                     break;
               }
               break;
            case kTE_TAB:
               switch (parm1) {
                  case Id1:
                     fTxt2->SetFocus();
                     break;
                  case Id2:
                     fTxt3->SetFocus();
                     break;
                  case Id3:
                     fTxt1->SetFocus();
                     break;
                  case Id4:
                     fTxt5->SetFocus();
                     break;
                  case Id5:
                     fTxt4->SetFocus();
                     break;
               }
               break;
            default:
               break;
         }
         break;

      default:
         break;
   }
   return kTRUE;
}

