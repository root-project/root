// Author: Sergey Linev, GSI   27/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TQt6GedEditor.h"

#include "TCanvas.h"
#include "TROOT.h"
#include "TColor.h"
#include "TQt6Canvas.h"
#include "QCanvasWidget.h"

#include "TClass.h"
#include "TBaseClass.h"
#include "TMethod.h"
#include "TMethodCall.h"

#include "TAttMarker.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttText.h"

#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QComboBox>
#include <QSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QColorDialog>
#include <QDoubleSpinBox>
#include <QGroupBox>

#include <memory>

using namespace ROOT::Experimental;

/** \class TQt6GedEditor
    \ingroup qt6canvas
    \brief TVirtualPadEditor ABI implementation for Qt6
*/

TQt6GedEditor::TQt6GedEditor(TCanvas *c)
{
   SetCanvas(c);
}

TQt6GedEditor::~TQt6GedEditor()
{
   Hide();
}


void TQt6GedEditor::SetCanvas(TCanvas *newcan)
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

void TQt6GedEditor::ConnectToCanvas(TCanvas *c)
{
   c->Connect("Selected(TVirtualPad*,TObject*,Int_t)", "ROOT::Experimental::TQt6GedEditor", this,
              "SetModel(TVirtualPad*,TObject*,Int_t)");
}

////////////////////////////////////////////////////////////////////////////////
/// Disconnect this editor from the Selected signal of fCanvas.

void TQt6GedEditor::DisconnectFromCanvas()
{
   if (fCanvas)
      Disconnect(fCanvas, "Selected(TVirtualPad*,TObject*,Int_t)", this, "SetModel(TVirtualPad*,TObject*,Int_t)");
}

////////////////////////////////////////////////////////////////////////////////
/// Activate object editors according to the selected object.

void TQt6GedEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t event)
{
   if (event != kButton1Down)
      return;

   auto prev = fModel;

   fPad = pad;
   fModel = obj ? obj : pad;

   if (fModel != prev)
      FillDialogsElements();
}

void TQt6GedEditor::Show()
{
   if (gPad)
      SetCanvas(gPad->GetCanvas());

   if (fCanvas && fGlobal)
      SetModel(fCanvas->GetClickSelectedPad(), fCanvas->GetClickSelected(), kButton1Down);

   if (!gROOT->GetListOfCleanups()->FindObject(this))
      gROOT->GetListOfCleanups()->Add(this);


   fDialog = new QDialog;
   fDialog->setWindowTitle("Edit Attributes");
   fDialog->setModal(false);

   auto imp = dynamic_cast<TQt6Canvas *>(fCanvas->GetCanvasImp());
   if (imp) {
     auto widget = imp->GetCanvasWidget();
     fDialog->resize(200, widget->height());
     QPoint pos = widget->mapToGlobal(QPoint(0, 0));
     fDialog->move(pos.x() - fDialog->width(), pos.y());
   }

   auto mainLayout = new QVBoxLayout(fDialog);
   fFormLayout = new QFormLayout();

   mainLayout->addLayout(fFormLayout);


   // --- Dialog Buttons (OK / Cancel) ---
   QHBoxLayout *buttonLayout = new QHBoxLayout();
   QPushButton *okButton = new QPushButton("OK");
   QPushButton *cancelButton = new QPushButton("Cancel");
   buttonLayout->addStretch();
   buttonLayout->addWidget(okButton);
   buttonLayout->addWidget(cancelButton);
   mainLayout->addLayout(buttonLayout);

   QObject::connect(okButton, &QPushButton::clicked, fDialog, &QDialog::accept);
   QObject::connect(cancelButton, &QPushButton::clicked, fDialog, &QDialog::reject);

   fDialog->setAttribute(Qt::WA_DeleteOnClose);

   QObject::connect(fDialog, &QDialog::finished, [this]([[maybe_unused]] int result) {
      fDialog = nullptr;
      fFormLayout = nullptr;
   });

   if (fModel)
      FillDialogsElements();

   fDialog->show();
}

void TQt6GedEditor::AddHLine(QFormLayout *f, const char *lbl)
{
   QWidget *container = new QWidget(fDialog);
   QHBoxLayout *layout = new QHBoxLayout(container);
   layout->setContentsMargins(0, 5, 0, 5);

   QFrame *leftLine = new QFrame(fDialog);
   leftLine->setFrameShape(QFrame::HLine);

   QLabel *label = new QLabel(lbl, fDialog);

   QFrame *rightLine = new QFrame(fDialog);
   rightLine->setFrameShape(QFrame::HLine);

   layout->addWidget(leftLine, 1);  // Stretch factor 1
   layout->addWidget(label, 0);     // Fits content tightly
   layout->addWidget(rightLine, 4); // Stret

   f->addRow(container);
}

void TQt6GedEditor::AddColorElements(int colindx, QFormLayout *layout, std::function<void(int)> callback)
{
   TColor *rootColor = gROOT->GetColor(colindx);
   QColor initialColor = Qt::black;
   int initialAlpha255 = 255; // Default fully opaque

   if (rootColor) {
      initialColor = QColor(rootColor->GetRed() * 255, rootColor->GetGreen() * 255, rootColor->GetBlue() * 255);
      initialAlpha255 = static_cast<int>(rootColor->GetAlpha() * 255);
   }
   initialColor.setAlpha(initialAlpha255);

   auto colorButton = new QPushButton(fDialog);
   colorButton->setFixedWidth(80);

   QSlider *alphaSlider = new QSlider(Qt::Horizontal);
   alphaSlider->setRange(0, 255);
   alphaSlider->setValue(initialAlpha255);

   // instance will be deleted when last lambda is removed
   auto selectedColor = std::make_shared<QColor>(initialColor);

   auto updateColorElements = [colorButton, selectedColor, callback]() {
      QString qss = QString("background-color: rgba(%1, %2, %3, %4); border: 1px solid gray;")
                        .arg(selectedColor->red())
                        .arg(selectedColor->green())
                        .arg(selectedColor->blue())
                        .arg(selectedColor->alpha() / 255.0);
      colorButton->setStyleSheet(qss);
      Color_t newColorIdx = TColor::GetColor(selectedColor->red(),
                                             selectedColor->green(),
                                             selectedColor->blue(),
                                             selectedColor->alpha() / 255.0);
      callback(newColorIdx);
   };

   updateColorElements();

   QObject::connect(colorButton, &QPushButton::clicked, [selectedColor, updateColorElements]() {
      QColor col = QColorDialog::getColor(*selectedColor, nullptr, "Select Color");
      if (col.isValid()) {
         selectedColor->setRed(col.red());
         selectedColor->setGreen(col.green());
         selectedColor->setBlue(col.blue());
         updateColorElements();
      }
   });

   // --- Slider Shift Connection ---
   QObject::connect(alphaSlider, &QSlider::valueChanged, [selectedColor, updateColorElements](int value) {
      selectedColor->setAlpha(value);
      updateColorElements();
   });

   layout->addRow("Color:", colorButton);

   layout->addRow("Opacity:", alphaSlider);
}

void TQt6GedEditor::ModifiedPad()
{
   auto pad = fPad;
   if (!pad)
      pad = fCanvas;
   if (pad)
      pad->ModifiedUpdate();
}

void TQt6GedEditor::AddTAttLine(TAttLine *attline)
{
   AddHLine(fFormLayout, "TAttLine");

   AddColorElements(attline->GetLineColor(), fFormLayout, [attline, this](int colindx) {
      attline->SetLineColor(colindx);
      ModifiedPad();
   });

   QComboBox *styleCombo = new QComboBox();
   styleCombo->addItem("None (0)", 0);
   styleCombo->addItem("Solid (1)", 1);
   styleCombo->addItem("Dashed (2)", 2);
   styleCombo->addItem("Dotted (3)", 3);
   styleCombo->addItem("Dash-Dot (4)", 4);
   styleCombo->addItem("Dash-Dot (5)", 5);
   styleCombo->addItem("Dash-Dot-Dot-Dot (6)", 6);
   styleCombo->addItem("Dashed medium (7)", 7);
   styleCombo->addItem("Dash-Dot-Dot (8)", 8);
   styleCombo->addItem("Dashed long (9)", 9);
   styleCombo->addItem("Dash-Dot long (10)", 10);

   // Find and set current style
   int currentStyle = attline->GetLineStyle();
   int styleIdx = styleCombo->findData(currentStyle);
   if (styleIdx != -1)
      styleCombo->setCurrentIndex(styleIdx);
   else
      styleCombo->addItem(QString("Custom (%1)").arg(currentStyle), currentStyle);

    QObject::connect(styleCombo, &QComboBox::currentIndexChanged, [this, attline, styleCombo](int) {
      attline->SetLineStyle(styleCombo->currentData().toInt());
      ModifiedPad();
   });

   fFormLayout->addRow("Style:", styleCombo);

   QSpinBox *widthSpin = new QSpinBox();
   widthSpin->setRange(1, 20);
   widthSpin->setValue(attline->GetLineWidth());

   QObject::connect(widthSpin, &QSpinBox::valueChanged, [this, attline](int v) {
      attline->SetLineWidth(v);
      ModifiedPad();
   });

   fFormLayout->addRow("Width:", widthSpin);
}

void TQt6GedEditor::AddTAttFill(TAttFill *attfill)
{
   AddHLine(fFormLayout, "TAttFill");

   AddColorElements(attfill->GetFillColor(), fFormLayout, [attfill, this](int colindx) {
      attfill->SetFillColor(colindx);
      ModifiedPad();
   });

   QComboBox *styleCombo = new QComboBox();
   styleCombo->addItem("None (0)", 0);
   styleCombo->addItem("Solid (1001)", 1001);
   for (int s = 3001; s <= 3025; ++s)
      styleCombo->addItem(QString("Style %1").arg(s), s);
   for (int s = 3144; s <= 3944; s += 100)
      styleCombo->addItem(QString("Style %1").arg(s), s);
   for (int s = 3305; s <= 3395; s += 10)
      styleCombo->addItem(QString("Style %1").arg(s), s);
   for (int s = 3350; s <= 3359; s += 1)
      styleCombo->addItem(QString("Style %1").arg(s), s);
   for (int s = 3409; s <= 3490; s += 9)
      styleCombo->addItem(QString("Style %1").arg(s), s);
   for (int s = 3609; s <= 3690; s += 9)
      styleCombo->addItem(QString("Style %1").arg(s), s);

   // Find and set current style
   int currentStyle = attfill->GetFillStyle();
   int styleIdx = styleCombo->findData(currentStyle);
   if (styleIdx != -1)
      styleCombo->setCurrentIndex(styleIdx);
   else
      styleCombo->addItem(QString("Style %1").arg(currentStyle), currentStyle);

   QObject::connect(styleCombo, &QComboBox::currentIndexChanged, [this, attfill, styleCombo](int) {
      attfill->SetFillStyle(styleCombo->currentData().toInt());
      ModifiedPad();
   });

   fFormLayout->addRow("Style:", styleCombo);
}


class CustomDoubleSpinBox : public QDoubleSpinBox {
protected:
    QString textFromValue(double value) const override {
        if (value == 0) return "Default";
        return QDoubleSpinBox::textFromValue(value);
    }

    double valueFromText(const QString &text) const override {
        if (text == "Default") return 0.;
        return QDoubleSpinBox::valueFromText(text);
    }
};

class CustomSpinBox : public QSpinBox {
protected:
    QString textFromValue(int value) const override {
        if (value == 0) return "Default";
        return QSpinBox::textFromValue(value);
    }

    int valueFromText(const QString &text) const override {
        if (text == "Default") return 0;
        return QSpinBox::valueFromText(text);
    }
};


void TQt6GedEditor::AddTAttText(TAttText *atttext)
{
   AddHLine(fFormLayout, "TAttText");

   AddColorElements(atttext->GetTextColor(), fFormLayout, [atttext, this](int colindx) {
      atttext->SetTextColor(colindx);
      ModifiedPad();
   });

   QComboBox *fontCombo = new QComboBox();
   fontCombo->addItem("1. Times italic", 1);
   fontCombo->addItem("2. Times bold", 2);
   fontCombo->addItem("3. Times bold italic", 3);
   fontCombo->addItem("4. Helvetica", 4);
   fontCombo->addItem("5. Helvetica italic", 5);
   fontCombo->addItem("6. Helvetica bold", 6);
   fontCombo->addItem("7. Helvetica bold italic", 7);
   fontCombo->addItem("8. Courier", 8);
   fontCombo->addItem("9. Courier italic", 9);
   fontCombo->addItem("10. Courier bold", 10);
   fontCombo->addItem("11. Courier bold italic", 11);
   fontCombo->addItem("12. Symbol", 12);
   fontCombo->addItem("13. Times", 13);
   fontCombo->addItem("14. Wingdings", 14);
   fontCombo->addItem("15. Symbol italic", 15);

   // Find and set current style
   int currentPrec = atttext->GetTextFont() % 10;
   int currentFont = atttext->GetTextFont() / 10;
   int styleIdx = fontCombo->findData(currentFont);
   if (styleIdx >= 0)
      fontCombo->setCurrentIndex(styleIdx);
   else
      fontCombo->addItem(QString("Font %1").arg(currentFont), currentFont);

   QObject::connect(fontCombo, &QComboBox::currentIndexChanged, [this, atttext, fontCombo, currentPrec](int) {
      atttext->SetTextFont(fontCombo->currentData().toInt() * 10 + currentPrec);
      ModifiedPad();
   });

   fFormLayout->addRow("Font:", fontCombo);

   // QDoubleSpinBox* floatSpinBox = nullptr;
   //QSpinBox *intSpinBox = nullptr;

   if (currentPrec == 2) {
      auto floatSpinBox = new CustomDoubleSpinBox();
      floatSpinBox->setRange(0.0, 1.0);   // Set your minimum and maximum limits
      floatSpinBox->setSingleStep(0.01);   // Set step size to 00.1
      floatSpinBox->setDecimals(3);        // Force it to show exactly 3 decimal places
      floatSpinBox->setValue(atttext->GetTextSize());

      QObject::connect(floatSpinBox, &QDoubleSpinBox::valueChanged, [this, atttext](double v) {
         atttext->SetTextSize(v);
         ModifiedPad();
      });
      fFormLayout->addRow("Size:", floatSpinBox);

   } else {
      auto intSpinBox = new CustomSpinBox();
      intSpinBox->setRange(0, 128);
      intSpinBox->setValue(atttext->GetTextSize());

      QObject::connect(intSpinBox, &QSpinBox::valueChanged, [this, atttext](int v) {
         atttext->SetTextSize(v);
         ModifiedPad();
      });

      fFormLayout->addRow("Size:", intSpinBox);
   }

   QComboBox *alignCombo = new QComboBox();
   alignCombo->addItem("11. Left Bottom", 11);
   alignCombo->addItem("12. Left Center", 12);
   alignCombo->addItem("13. Left Top", 13);
   alignCombo->addItem("21. Middle Bottom", 21);
   alignCombo->addItem("22. Middle Center", 22);
   alignCombo->addItem("23. Middle Top", 23);
   alignCombo->addItem("31. Right Bottom", 31);
   alignCombo->addItem("32. Right Center", 32);
   alignCombo->addItem("33. Right Top", 33);

   // Find and set current style
   int alignIdx = alignCombo->findData(atttext->GetTextAlign());
   if (alignIdx < 0)
      alignIdx = alignCombo->findData(11);
   alignCombo->setCurrentIndex(alignIdx);

   QObject::connect(alignCombo, &QComboBox::currentIndexChanged, [this, atttext, alignCombo](int) {
      atttext->SetTextAlign(alignCombo->currentData().toInt());
      ModifiedPad();
   });

   fFormLayout->addRow("Align:", alignCombo);
}



void TQt6GedEditor::AddTAttMarker(TAttMarker *attmarker)
{
   AddHLine(fFormLayout, "TAttMarker");

   AddColorElements(attmarker->GetMarkerColor(), fFormLayout, [attmarker, this](int colindx) {
      attmarker->SetMarkerColor(colindx);
      ModifiedPad();
   });

   QComboBox *styleCombo = new QComboBox();
   for (int s = 1; s <= 49; ++s)
      styleCombo->addItem(QString("Style %1").arg(s), s);

   // Find and set current style
   int currentStyle = attmarker->GetMarkerStyle();
   int styleIdx = styleCombo->findData(currentStyle);
   if (styleIdx != -1)
      styleCombo->setCurrentIndex(styleIdx);
   else
      styleCombo->addItem(QString("Style %1").arg(currentStyle), currentStyle);

   QObject::connect(styleCombo, &QComboBox::currentIndexChanged, [this, attmarker, styleCombo](int) {
      attmarker->SetMarkerStyle(styleCombo->currentData().toInt());
      ModifiedPad();
   });

   fFormLayout->addRow("Style:", styleCombo);

   QDoubleSpinBox *floatSpinBox = new QDoubleSpinBox();
   floatSpinBox->setRange(0.0, 100.0); // Set your minimum and maximum limits
   floatSpinBox->setSingleStep(1);     // Set step size to 1
   floatSpinBox->setDecimals(1);       // Force it to show exactly 1 decimal place (e.g., 1.5)
   floatSpinBox->setValue(attmarker->GetMarkerSize());
   fFormLayout->addRow("Size:", floatSpinBox);

   QObject::connect(floatSpinBox, &QDoubleSpinBox::valueChanged, [this, attmarker](double v) {
      attmarker->SetMarkerSize(v);
      ModifiedPad();
   });
}

void TQt6GedEditor::FillGed(TClass *cl)
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


void TQt6GedEditor::FillDialogsElements()
{
   if (!fFormLayout || !fModel)
      return;

   // first delete all elements
   while (fFormLayout->count() > 0) {
      // Always take from index 0 or use a reverse loop
      auto item = fFormLayout->takeAt(0);

      if (QWidget *widget = item->widget())
         widget->deleteLater(); // Safely deletes the widget

      delete item; // Deletes the layout item wrapper
   }

   FillGed(fModel->IsA());

//   auto attmarker = dynamic_cast<TAttMarker *>(fModel);
//   if (attmarker)
//      AddTAttMarker(attmarker);
}

void TQt6GedEditor::Hide()
{
   gROOT->GetListOfCleanups()->Remove(this);
   if (fDialog) {
      fDialog->close();
      fDialog = nullptr;
      fFormLayout = nullptr;
   }
}

void TQt6GedEditor::RecursiveRemove(TObject* obj)
{
   if (obj == fModel) {
      SetModel(fPad, fPad, kButton1Down);
   } else if (obj == fPad) {
      SetModel(fCanvas, fCanvas, kButton1Down);
   } else if (obj == fCanvas)
      Hide();
}
