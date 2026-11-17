// Author: Sergey Linev, GSI   30/10/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGtk4GedEditor
#define ROOT_TGtk4GedEditor

#include "TVirtualPadEditor.h"
#include "TQObject.h"

#include <functional>

class TVirtualPad;
class TAttMarker;
class TAttText;
class TAttLine;
class TAttFill;
class TClass;

class Gtk4MethodDialog;

namespace Gtk {
   class DropDown;
   class SpinButton;
   class ColorDialogButton;
}


namespace ROOT {
namespace Experimental {

class TGtk4GedEditor : public TVirtualPadEditor, public TObject, public TQObject {

   protected:

      TCanvas *fCanvas = nullptr;
      TVirtualPad *fPad = nullptr;
      TObject *fModel = nullptr;
      Bool_t fGlobal = kTRUE;

      Gtk4MethodDialog *fDialog = nullptr;

      TAttLine *fAttLine = nullptr;
      TAttFill *fAttFill = nullptr;
      TAttText *fAttText = nullptr;
      TAttMarker *fAttMarker = nullptr;

      Gtk::ColorDialogButton* m_line_color = nullptr;
      Gtk::DropDown *m_line_style = nullptr;
      Gtk::SpinButton *m_line_width = nullptr;

      Gtk::ColorDialogButton* m_fill_color = nullptr;
      Gtk::ColorDialogButton* m_text_color = nullptr;
      Gtk::ColorDialogButton* m_marker_color = nullptr;

      Int_t GetRootColor(Gtk::ColorDialogButton* btn);

      void FillDialogsElements();

      void ModifiedPad();

      void FillGed(TClass *cl);

      void on_ok_button_clicked();

   public:

      TGtk4GedEditor(TCanvas *c = nullptr);
      virtual ~TGtk4GedEditor();

      Bool_t   IsGlobal() const override { return fGlobal; }
      void     SetGlobal(Bool_t on) override { fGlobal = on; }

      TCanvas* GetCanvas() const override { return fCanvas; }

      void ConnectToCanvas(TCanvas *c);
      void DisconnectFromCanvas();
      void SetCanvas(TCanvas *c);

      void Show() override;
      void Hide() override;

      void RecursiveRemove(TObject* obj) override;

      virtual void SetModel(TVirtualPad* pad, TObject* obj, Int_t event);

      // specific editors
      void AddTAttLine(TAttLine *);
      void AddTAttFill(TAttFill *);
      void AddTAttText(TAttText *);
      void AddTAttMarker(TAttMarker *);

   ClassDefOverride(TGtk4GedEditor, 0) // Implementation for Ged Editor with GTK4
};

} // namespace Experimental
} // namespace ROOT

#endif
