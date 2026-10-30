// Author: Sergey Linev, GSI   30/10/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGtk4GedEditor.h"

#include "Gtk4MethodDialog.h"
#include "TGtk4Canvas.h"

#include "TCanvas.h"
#include "TROOT.h"
#include "TColor.h"

#include "TClass.h"
#include "TBaseClass.h"
#include "TMethod.h"
#include "TMethodCall.h"

#include "TAttMarker.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttText.h"



#include <memory>

using namespace ROOT::Experimental;

/** \class TGtk4GedEditor
    \ingroup gtk4canvas
    \brief TVirtualPadEditor ABI implementation for GTK4
*/

TGtk4GedEditor::TGtk4GedEditor(TCanvas *c)
{
   SetCanvas(c);
}

TGtk4GedEditor::~TGtk4GedEditor()
{
}


void TGtk4GedEditor::SetCanvas(TCanvas *newcan)
{
   if (fCanvas == newcan) return;

   DisconnectFromCanvas();
   fCanvas = newcan;

   if (!newcan) return;

   // SetWindowName(Form("%s_Editor", fCanvas->GetName()));
   fPad = fCanvas->GetSelectedPad();
   if (!fPad) fPad = fCanvas;
   ConnectToCanvas(fCanvas);
}


////////////////////////////////////////////////////////////////////////////////
/// Connect this editor to the Selected signal of canvas 'c'.

void TGtk4GedEditor::ConnectToCanvas(TCanvas *c)
{
   c->Connect("Selected(TVirtualPad*,TObject*,Int_t)", "ROOT::Experimental::TGtk4GedEditor", this,
              "SetModel(TVirtualPad*,TObject*,Int_t)");
}

////////////////////////////////////////////////////////////////////////////////
/// Disconnect this editor from the Selected signal of fCanvas.

void TGtk4GedEditor::DisconnectFromCanvas()
{
   if (fCanvas)
      Disconnect(fCanvas, "Selected(TVirtualPad*,TObject*,Int_t)", this, "SetModel(TVirtualPad*,TObject*,Int_t)");
}

////////////////////////////////////////////////////////////////////////////////
/// Activate object editors according to the selected object.

void TGtk4GedEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t event)
{
   if (event != kButton1Down)
      return;

   auto prev = fModel;

   fPad = pad;
   fModel = obj ? obj : pad;

   if (fModel != prev)
      FillDialogsElements();
}

void TGtk4GedEditor::Show()
{
   if (gPad)
      SetCanvas(gPad->GetCanvas());

   if (fCanvas && fGlobal)
      SetModel(fCanvas->GetClickSelectedPad(), fCanvas->GetClickSelected(), kButton1Down);

   if (!gROOT->GetListOfCleanups()->FindObject(this))
      gROOT->GetListOfCleanups()->Add(this);

   fDialog = new Gtk4MethodDialog(300, 300);

   /* reposition dialog
   auto imp = dynamic_cast<TGtk4Canvas *>(fCanvas->GetCanvasImp());
   if (imp) {
     auto widget = imp->GetCanvasWidget();
     fDialog->resize(200, widget->height());
     QPoint pos = widget->mapToGlobal(QPoint(0, 0));
     fDialog->move(pos.x() - fDialog->width(), pos.y());
   }
   */

   if (fModel)
      FillDialogsElements();
}


void TGtk4GedEditor::ModifiedPad()
{
   auto pad = fPad;
   if (!pad)
      pad = fCanvas;
   if (pad)
      pad->ModifiedUpdate();
}

void TGtk4GedEditor::AddTAttLine(TAttLine *attline)
{
   m_line_color = fDialog->addColorInput("Line col", attline->GetLineColor());

   auto style_list = Gtk::StringList::create({"None", "Style 1", "Style 2", "Style 3", "Style 4", "Style 5", "Style 6", "Style 7", "Style 8", "Style 9"});
   m_line_style = Gtk::make_managed<Gtk::DropDown>(style_list);
   m_line_style->set_selected(attline->GetLineStyle());

   fDialog->addLine("Line style", m_line_style);

   auto adjustment = Gtk::Adjustment::create(/* value */ attline->GetLineWidth(), /* lower */ 1, /* upper */ 20,
                                            /* step_increment */ 1, /* page_increment */ 5.0);

   m_line_width = Gtk::make_managed<Gtk::SpinButton>(adjustment);
   m_line_width->set_digits(0); // 0 decimal places -> integer display

   fDialog->addLine("Line width", m_line_width);

   fAttLine = attline;
}

void TGtk4GedEditor::AddTAttFill(TAttFill *attfill)
{
   m_fill_color = fDialog->addColorInput("Fill col", attfill->GetFillColor());

   fAttFill = attfill;
}

void TGtk4GedEditor::AddTAttText(TAttText *atttext)
{
   m_text_color = fDialog->addColorInput("Text col", atttext->GetTextColor());

   fAttText = atttext;
}

void TGtk4GedEditor::AddTAttMarker(TAttMarker *attmark)
{
   m_marker_color = fDialog->addColorInput("Marker col", attmark->GetMarkerColor());

   fAttMarker = attmark;
}

void TGtk4GedEditor::FillGed(TClass *cl)
{
   TString method_name = TString::Format("Add%s", cl->GetName());

   auto mtd = IsA()->GetMethodAny(method_name.Data());

   if (mtd) {
      void *obj = fModel->IsA()->DynamicCast(cl, fModel);
      auto sarg = TString::Format("(%s*)0x%zx", cl->GetName(), (size_t)obj);
      TMethodCall call(IsA(), method_name, sarg.Data());
      call.Execute(this);
   }

   auto lst = cl->GetListOfBases();

   TIter iter(lst);
   while (auto base = (TBaseClass*) iter())
      FillGed(base->GetClassPointer());
}

void TGtk4GedEditor::FillDialogsElements()
{
   if (!fModel || !fDialog)
      return;

   // first delete all previous elements !!!!
   fDialog->clearAllElements();

   fAttLine = nullptr;
   fAttFill = nullptr;
   fAttText = nullptr;
   fAttMarker = nullptr;
   m_line_color = m_fill_color = m_text_color = m_marker_color = nullptr;
   m_line_style = nullptr;
   m_line_width = nullptr;

   FillGed(fModel->IsA());

   auto okBtn = fDialog->addOkCancelButtons();

   okBtn->signal_clicked().connect(sigc::mem_fun(*this, &TGtk4GedEditor::on_ok_button_clicked));

   fDialog->present();
}

Int_t TGtk4GedEditor::GetRootColor(Gtk::ColorDialogButton* btn)
{
   auto selected_color = btn->get_rgba();
   return TColor::GetColor(selected_color.get_red(),
                           selected_color.get_green(),
                           selected_color.get_blue(),
                           selected_color.get_alpha());
}


void TGtk4GedEditor::on_ok_button_clicked()
{
   if (fAttLine) {
      fAttLine->SetLineColor(GetRootColor(m_line_color));
      fAttLine->SetLineStyle(m_line_style->get_selected());
      fAttLine->SetLineWidth(m_line_width->get_value());
   }

   if (fAttFill) {
      fAttFill->SetFillColor(GetRootColor(m_fill_color));
   }

   if (fAttText) {
      fAttText->SetTextColor(GetRootColor(m_text_color));
   }
   if (fAttMarker) {
      fAttMarker->SetMarkerColor(GetRootColor(m_marker_color));
   }

   fDialog->close();
   fDialog = nullptr;

   if (gPad)
      gPad->ModifiedUpdate();

}


void TGtk4GedEditor::Hide()
{
   gROOT->GetListOfCleanups()->Remove(this);
   if (fDialog) {
      fDialog->close();
      fDialog = nullptr;
   }
}

void TGtk4GedEditor::RecursiveRemove(TObject* obj)
{
   if (obj == fModel) {
      SetModel(fPad, fPad, kButton1Down);
   } else if (obj == fPad) {
      SetModel(fCanvas, fCanvas, kButton1Down);
   } else if (obj == fCanvas)
      Hide();
}
