// Author: Sergey Linev  2/07/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class QRootContextMenu
    \ingroup qt6canvas

This class provides an interface to context-sensitive popup menus.
These menus pop up when the user hits the right mouse button, and
are destroyed when the menu pops downs.
*/


#include "QRootContextMenu.h"

#include "TROOT.h"
#include "TContextMenu.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TMethod.h"
#include "TDataMember.h"
#include "TToggle.h"
#include "TClassMenuItem.h"
#include "TAttText.h"
#include "TAttMarker.h"

#include "QRootMethodDialog.h"
#include "QPaintWidget.h"
#include "TQt6Canvas.h"

#include <QtCore/QSignalMapper>
#include <QMenu>
#include <QAction>
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

enum EContextMenu {
   kToggleStart       = 1000, // first id of toggle menu items
   kToggleListStart   = 2000, // first id of toggle list menu items
   kUserFunctionStart = 3000  // first id of user added functions/methods, etc...
};

////////////////////////////////////////////////////////////////////////////////
/// Create context menu.

QRootContextMenu::QRootContextMenu(TContextMenu *c, const char *)
    : QObject(), TObject(), TContextMenuImp(c)
{
   gROOT->GetListOfCleanups()->Add(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete a context menu.

QRootContextMenu::~QRootContextMenu()
{
   gROOT->GetListOfCleanups()->Remove(this);
   fTrash.Delete();
}

////////////////////////////////////////////////////////////////////////////////
/// Display context popup menu for currently selected object.

void QRootContextMenu::DisplayPopup(Int_t x, Int_t y)
{
   // add menu items to popup menu
   // CreateMenu(fContextMenu->GetSelectedObject());

   auto object = fContextMenu->GetSelectedObject();
   if (!object)
      return;

   fCustomArg.clear();
   fTrash.Delete();


   auto canv = dynamic_cast<TCanvas *>(fContextMenu->GetSelectedCanvas());
   auto canvimp = dynamic_cast<ROOT::Experimental::TQt6Canvas *>(canv->GetCanvasImp());
   auto widget = canvimp->GetPaintWidget();

   QPoint screenPos = widget->mapToGlobal(widget->rect().topLeft());

   QMenu menu;
   QSignalMapper map;

   QObject::connect(&map, &QSignalMapper::mappedInt,
                    this, &QRootContextMenu::executeMenu);

   // Add a title
   QString buffer = fContextMenu->CreatePopupTitle(object);
   addMenuAction(&menu, &map, buffer, -1, nullptr);
   menu.addSeparator();
   bool last_separ = true;

   int entry = 0, toggle = kToggleStart, togglelist = kToggleListStart;
   int userfunction = kUserFunctionStart;

   // Get list of menu items from the selected object's class
   TList *menuItemList = object->IsA()->GetMenuList();

   TIter nextItem(menuItemList);

   while (auto menuItem = (TClassMenuItem*) nextItem()) {
      switch (menuItem->GetType()) {
         case TClassMenuItem::kPopupSeparator: {
            if (!last_separ)
               menu.addSeparator();
            last_separ = true;
            break;
         }
         case TClassMenuItem::kPopupStandardList: {
            // Standard list of class methods. Rebuild from scratch.
            // Get linked list of objects menu items (i.e. member functions
            // with the token *MENU in their comment fields.
            TList *methodList = new TList;
            object->IsA()->GetMenuItems(methodList);

            TMethod *method;
            TClass  *classPtr = nullptr;
            TIter next(methodList);
            Bool_t needSep = kFALSE;

            while ((method = (TMethod*) next())) {
               if (classPtr != method->GetClass()) {
                  needSep = kTRUE;
                  classPtr = method->GetClass();
               }

               EMenuItemKind menuKind = method->IsMenuItem();
               TString last_component;

               switch (menuKind) {
                  case kMenuDialog:
                     // search for arguments to the MENU statement
                     if (needSep) {
                        menu.addSeparator();
                        needSep = kFALSE;
                     }
                     addMenuAction(&menu, &map, method->GetName(), entry++, method);
                     break;
                  case kMenuSubMenu:
                     if (auto m = method->FindDataMember()) {
                        if (needSep) {
                           menu.addSeparator();
                           needSep = kFALSE;
                        }

                        if (m->GetterMethod()) {

                           QMenu *r = menu.addMenu(method->GetName());
                           TIter nxt(m->GetOptions());
                           while (auto it = (TOptionListItem*) nxt()) {
                              const char *name = it->fOptName;
                              Long_t val = it->fValue;

                              TToggle *t = new TToggle;
                              t->SetToggledObject(object, method);
                              t->SetOnValue(val);
                              fTrash.Add(t);

                              auto act = addMenuAction(r, &map, name, togglelist++, t);
                              act->setCheckable(true);
                              if (t->GetState())
                                 act->setChecked(true);
                           }
                        } else {
                           addMenuAction(&menu, &map, method->GetName(), entry++, method);
                        }
                     }
                     break;

                  case kMenuToggle: {
                     if (needSep) {
                        menu.addSeparator();
                        needSep = kFALSE;
                     }

                     TToggle *t = new TToggle;
                     t->SetToggledObject(object, method);
                     t->SetOnValue(1);
                     fTrash.Add(t);

                     auto act = addMenuAction(&menu, &map, method->GetName(), toggle++, t);
                     act->setCheckable(true);
                     if (t->GetState())
                        act->setChecked(true);
                     break;
                  }
                  default:
                     break;
               }
            }
            delete methodList;
         }
         break;
         case TClassMenuItem::kPopupUserFunction: {
            const char* menuItemTitle = menuItem->GetTitle();
            if (menuItem->IsToggle()) {
               TMethod* method = object->IsA()->GetMethodWithPrototype(menuItem->GetFunctionName(),menuItem->GetArgs());
               if (method) {
                  TToggle *t = new TToggle;
                  t->SetToggledObject(object, method);
                  t->SetOnValue(1);
                  fTrash.Add(t);

                  if (strlen(menuItemTitle)==0)
                     menuItemTitle = method->GetName();
                  auto act = addMenuAction(&menu, &map, menuItemTitle, toggle++, t);
                  act->setCheckable(true);
                  if (t->GetState())
                     act->setChecked(true);
               }
            } else {
               if (strlen(menuItemTitle)==0)
                  menuItemTitle = menuItem->GetFunctionName();
               addMenuAction(&menu, &map, menuItemTitle, userfunction++, menuItem);
            }
            break;
         }

         default:
            break;
      }
   }

   menu.exec(screenPos + QPoint(x, y));
}

////////////////////////////////////////////////////////////////////////////////
/// Create dialog object with OK and Cancel buttons. This dialog
/// prompts for the arguments of "method".

void QRootContextMenu::Dialog(TObject *object, TMethod *method)
{
   Dialog(object, (TFunction *)method);
}

////////////////////////////////////////////////////////////////////////////////
/// Create dialog object with OK and Cancel buttons. This dialog
/// prompts for the arguments of "function".
/// function may be a global function or a method

void QRootContextMenu::Dialog(TObject * object, TFunction * func)
{
   QRootMethodDialog dlg;
   dlg.methodDialog(fContextMenu, object, func);
}

////////////////////////////////////////////////////////////////////////////////
/// Handle remove of some ROOT objects

void QRootContextMenu::RecursiveRemove(TObject *obj)
{
   if (obj == fContextMenu->GetSelectedCanvas())
      fContextMenu->SetCanvas(nullptr);
   if (obj == fContextMenu->GetSelectedPad())
      fContextMenu->SetPad(nullptr);
   if (obj == fContextMenu->GetSelectedObject()) {
      // if the object being deleted is the one selected,
      // ungrab the mouse pointer and terminate (close) the menu
      fContextMenu->SetObject(nullptr);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Register menu action in signal map

QAction* QRootContextMenu::addMenuAction(QMenu* menu, QSignalMapper *map, const QString& text, int id, void *arg)
{
   bool enabled = true;

   QAction* act = new QAction(text, menu);

   if (!enabled)
      if ((text.compare("DrawClone") == 0) || (text.compare("DrawClass") == 0) || (text.compare("Inspect") == 0) ||
          (text.compare("SetShowProjectionX") == 0) || (text.compare("SetShowProjectionY") == 0) ||
          (text.compare("DrawPanel") == 0) || (text.compare("FitPanel") == 0))
         act->setEnabled(false);

   QObject::connect(act, &QAction::triggered, [id, map]() {
      map->mappedInt(id);
   });

   menu->addAction(act);
   map->setMapping(act, id);

   fCustomArg[id] = arg;

   return act;
}

////////////////////////////////////////////////////////////////////////////////
/// Add color elements to attributes editor dialog

void QRootContextMenu::AddColorElements(int colindx, QFormLayout *layout)
{
   TColor *rootColor = gROOT->GetColor(colindx);
   QColor initialColor = Qt::black;
   int initialAlpha255 = 255; // Default fully opaque

   if (rootColor) {
      initialColor = QColor(rootColor->GetRed() * 255, rootColor->GetGreen() * 255, rootColor->GetBlue() * 255);
      initialAlpha255 = static_cast<int>(rootColor->GetAlpha() * 255);
   }
   initialColor.setAlpha(initialAlpha255);

   fColorButton = new QPushButton();
   fColorButton->setFixedWidth(80);

   QSlider *alphaSlider = new QSlider(Qt::Horizontal);
   alphaSlider->setRange(0, 255);
   alphaSlider->setValue(initialAlpha255);

   // Visual preview of current color
   fSelectedColor = initialColor;
   UpdateColorElements();

   QObject::connect(fColorButton, &QPushButton::clicked, [&, this]() {
      QColor col = QColorDialog::getColor(fSelectedColor, nullptr, "Select Color");
      if (col.isValid()) {
         fSelectedColor.setRed(col.red());
         fSelectedColor.setGreen(col.green());
         fSelectedColor.setBlue(col.blue());
         UpdateColorElements();
      }
   });

   // --- Slider Shift Connection ---
   QObject::connect(alphaSlider, &QSlider::valueChanged, [&](int value) {
      fSelectedColor.setAlpha(value);
      UpdateColorElements();
   });

   layout->addRow("Color:", fColorButton);

   layout->addRow("Opacity:", alphaSlider);
}

////////////////////////////////////////////////////////////////////////////////
/// Update color button with currently selected color

void QRootContextMenu::UpdateColorElements()
{
   QString qss = QString("background-color: rgba(%1, %2, %3, %4); border: 1px solid gray;")
                     .arg(fSelectedColor.red())
                     .arg(fSelectedColor.green())
                     .arg(fSelectedColor.blue())
                     .arg(fSelectedColor.alpha() / 255.0);
   fColorButton->setStyleSheet(qss);
}

////////////////////////////////////////////////////////////////////////////////
/// Start TAttLine editor

void QRootContextMenu::SetLineAttributesDialog()
{
   auto attline = dynamic_cast<TAttLine *>(fContextMenu->GetSelectedObject());
   if (!attline)
      return;

   QDialog dialog;
   dialog.setWindowTitle("Edit Line Attributes");
   dialog.setModal(true);

   QVBoxLayout *mainLayout = new QVBoxLayout(&dialog);
   QFormLayout *formLayout = new QFormLayout();

   // --- Color Selector ---
   AddColorElements(attline->GetLineColor(), formLayout);

   // --- Line Style Selector ---
   // ROOT Styles: 1=Solid, 2=Dashed, 3=Dotted, 4=Dash-Dot
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

   formLayout->addRow("Style:", styleCombo);

   // --- Line Width Selector ---
   QSpinBox *widthSpin = new QSpinBox();
   widthSpin->setRange(1, 20);
   widthSpin->setValue(attline->GetLineWidth());
   formLayout->addRow("Width:", widthSpin);

   mainLayout->addLayout(formLayout);

   // --- Dialog Buttons (OK / Cancel) ---
   QHBoxLayout *buttonLayout = new QHBoxLayout();
   QPushButton *okButton = new QPushButton("OK");
   QPushButton *cancelButton = new QPushButton("Cancel");
   buttonLayout->addStretch();
   buttonLayout->addWidget(okButton);
   buttonLayout->addWidget(cancelButton);
   mainLayout->addLayout(buttonLayout);

   QObject::connect(okButton, &QPushButton::clicked, &dialog, &QDialog::accept);
   QObject::connect(cancelButton, &QPushButton::clicked, &dialog, &QDialog::reject);

   // 2. Execute Dialog and apply properties if accepted
   if (dialog.exec() == QDialog::Accepted) {
      // Update ROOT Color Index
      Color_t newColorIdx = TColor::GetColor(fSelectedColor.red(),
                                             fSelectedColor.green(),
                                             fSelectedColor.blue(),
                                             fSelectedColor.alpha() / 255.0);
      attline->SetLineColor(newColorIdx);

      // Update Line Style
      Style_t newStyle = styleCombo->currentData().toInt();
      attline->SetLineStyle(newStyle);

      // Update Line Width
      Width_t newWidth = widthSpin->value();
      attline->SetLineWidth(newWidth);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Start TAttFill editor

void QRootContextMenu::SetFillAttributesDialog()
{
   auto attfill = dynamic_cast<TAttFill *>(fContextMenu->GetSelectedObject());
   if (!attfill)
      return;

   QDialog dialog;
   dialog.setWindowTitle("Edit Fill Attributes");
   dialog.setModal(true);

   QVBoxLayout *mainLayout = new QVBoxLayout(&dialog);
   QFormLayout *formLayout = new QFormLayout();

   // --- Color Selector ---
   AddColorElements(attfill->GetFillColor(), formLayout);

   // --- Fill Style Selector ---
   // ROOT Styles: 1=Solid, 2=Dashed, 3=Dotted, 4=Dash-Dot
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

   formLayout->addRow("Style:", styleCombo);

   mainLayout->addLayout(formLayout);

   // --- Dialog Buttons (OK / Cancel) ---
   QHBoxLayout *buttonLayout = new QHBoxLayout();
   QPushButton *okButton = new QPushButton("OK");
   QPushButton *cancelButton = new QPushButton("Cancel");
   buttonLayout->addStretch();
   buttonLayout->addWidget(okButton);
   buttonLayout->addWidget(cancelButton);
   mainLayout->addLayout(buttonLayout);

   QObject::connect(okButton, &QPushButton::clicked, &dialog, &QDialog::accept);
   QObject::connect(cancelButton, &QPushButton::clicked, &dialog, &QDialog::reject);

   // 2. Execute Dialog and apply properties if accepted
   if (dialog.exec() == QDialog::Accepted) {
      // Update ROOT Color Index
      Color_t newColorIdx = TColor::GetColor(fSelectedColor.red(),
                                             fSelectedColor.green(),
                                             fSelectedColor.blue(),
                                             fSelectedColor.alpha() / 255.0);
      attfill->SetFillColor(newColorIdx);

      // Update fill Style
      Style_t newStyle = styleCombo->currentData().toInt();
      attfill->SetFillStyle(newStyle);
   }
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


////////////////////////////////////////////////////////////////////////////////
/// Start TAttText editor

void QRootContextMenu::SetTextAttributesDialog()
{
   auto atttext = dynamic_cast<TAttText *>(fContextMenu->GetSelectedObject());
   if (!atttext)
      return;

   QDialog dialog;
   dialog.setWindowTitle("Edit Text Attributes");
   dialog.setModal(true);

   QVBoxLayout *mainLayout = new QVBoxLayout(&dialog);
   QFormLayout *formLayout = new QFormLayout();

   // --- Color Selector ---
   AddColorElements(atttext->GetTextColor(), formLayout);

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

   formLayout->addRow("Font:", fontCombo);

   QDoubleSpinBox* floatSpinBox = nullptr;
   QSpinBox *intSpinBox = nullptr;

   if (currentPrec == 2) {
      floatSpinBox = new CustomDoubleSpinBox();
      floatSpinBox->setRange(0.0, 1.0);   // Set your minimum and maximum limits
      floatSpinBox->setSingleStep(0.01);   // Set step size to 00.1
      floatSpinBox->setDecimals(3);        // Force it to show exactly 3 decimal places
      floatSpinBox->setValue(atttext->GetTextSize());
      formLayout->addRow("Size:", floatSpinBox);
   } else {
      intSpinBox = new CustomSpinBox();
      intSpinBox->setRange(0, 128);
      intSpinBox->setValue(atttext->GetTextSize());
      formLayout->addRow("Size:", intSpinBox);
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

   formLayout->addRow("Align:", alignCombo);

   mainLayout->addLayout(formLayout);

   // --- Dialog Buttons (OK / Cancel) ---
   QHBoxLayout *buttonLayout = new QHBoxLayout();
   QPushButton *okButton = new QPushButton("OK");
   QPushButton *cancelButton = new QPushButton("Cancel");
   buttonLayout->addStretch();
   buttonLayout->addWidget(okButton);
   buttonLayout->addWidget(cancelButton);
   mainLayout->addLayout(buttonLayout);

   QObject::connect(okButton, &QPushButton::clicked, &dialog, &QDialog::accept);
   QObject::connect(cancelButton, &QPushButton::clicked, &dialog, &QDialog::reject);

   // 2. Execute Dialog and apply properties if accepted
   if (dialog.exec() == QDialog::Accepted) {
      // Update ROOT Color Index
      Color_t newColorIdx = TColor::GetColor(fSelectedColor.red(),
                                             fSelectedColor.green(),
                                             fSelectedColor.blue(),
                                             fSelectedColor.alpha() / 255.0);
      atttext->SetTextColor(newColorIdx);

      // Update font Style
      Int_t newFont = fontCombo->currentData().toInt();
      atttext->SetTextFont(newFont * 10 + currentPrec);

      if (floatSpinBox)
         atttext->SetTextSize(floatSpinBox->value());
      else if (intSpinBox)
         atttext->SetTextSize(intSpinBox->value());

      atttext->SetTextAlign(alignCombo->currentData().toInt());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Start TAttMarker editor

void QRootContextMenu::SetMarkerAttributesDialog()
{
   auto attmarker = dynamic_cast<TAttMarker *>(fContextMenu->GetSelectedObject());
   if (!attmarker)
      return;

   QDialog dialog;
   dialog.setWindowTitle("Edit Marker Attributes");
   dialog.setModal(true);

   QVBoxLayout *mainLayout = new QVBoxLayout(&dialog);
   QFormLayout *formLayout = new QFormLayout();

   // --- Color Selector ---
   AddColorElements(attmarker->GetMarkerColor(), formLayout);

   // --- Marker Style Selector ---
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

   formLayout->addRow("Style:", styleCombo);


   QDoubleSpinBox* floatSpinBox = new QDoubleSpinBox();

   // 2. Configure its ranges and step parameters
   floatSpinBox->setRange(0.0, 10.0);   // Set your minimum and maximum limits
   floatSpinBox->setSingleStep(0.1);    // Set step size to 0.1
   floatSpinBox->setDecimals(1);        // Force it to show exactly 1 decimal place (e.g., 1.5)
   floatSpinBox->setValue(attmarker->GetMarkerSize());
   formLayout->addRow("Size:", floatSpinBox);

   mainLayout->addLayout(formLayout);

   // --- Dialog Buttons (OK / Cancel) ---
   QHBoxLayout *buttonLayout = new QHBoxLayout();
   QPushButton *okButton = new QPushButton("OK");
   QPushButton *cancelButton = new QPushButton("Cancel");
   buttonLayout->addStretch();
   buttonLayout->addWidget(okButton);
   buttonLayout->addWidget(cancelButton);
   mainLayout->addLayout(buttonLayout);

   QObject::connect(okButton, &QPushButton::clicked, &dialog, &QDialog::accept);
   QObject::connect(cancelButton, &QPushButton::clicked, &dialog, &QDialog::reject);

   // 2. Execute Dialog and apply properties if accepted
   if (dialog.exec() == QDialog::Accepted) {
      // Update ROOT Color Index
      Color_t newColorIdx = TColor::GetColor(fSelectedColor.red(),
                                             fSelectedColor.green(),
                                             fSelectedColor.blue(),
                                             fSelectedColor.alpha() / 255.0);
      attmarker->SetMarkerColor(newColorIdx);

      // Update Style
      Style_t newStyle = styleCombo->currentData().toInt();
      attmarker->SetMarkerStyle(newStyle);

      float newsize = floatSpinBox->value();
      attmarker->SetMarkerSize(newsize);
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Execute specified menu item

void QRootContextMenu::executeMenu(int id)
{
   if (id < 0)
      return;
   void *ud = fCustomArg[id];

   if (ud) {
      // retrieve the highlighted function
      TFunction *function = nullptr;
      if (id < kToggleStart) {
         TMethod *m = (TMethod *)ud;
         function = (TFunction *)m;
      } else if (id >= kToggleStart && id < kUserFunctionStart) {
         TToggle *t = (TToggle *)ud;
         TMethodCall *mc = (TMethodCall *)t->GetSetter();
         function = (TFunction *)mc->GetMethod();
      } else {
         TClassMenuItem *mi = (TClassMenuItem *)ud;
         function = gROOT->GetGlobalFunctionWithPrototype(mi->GetFunctionName());
      }
      if (function)
         fContextMenu->SetMethod(function);
   }

   if (id < kToggleStart) {
      auto m = (TMethod *) ud;

      if (!strcmp(m->GetName(), "SetLineAttributes")) {
         SetLineAttributesDialog();
      } else if (!strcmp(m->GetName(), "SetFillAttributes")) {
         SetFillAttributesDialog();
      } else if (!strcmp(m->GetName(), "SetTextAttributes")) {
         SetTextAttributesDialog();
      } else if (!strcmp(m->GetName(), "SetMarkerAttributes")) {
         SetMarkerAttributesDialog();
      } else {
         fContextMenu->Action(m);
      }
   } else if (id >= kToggleStart && id < kToggleListStart) {
      TToggle *t = (TToggle *) ud;
      fContextMenu->Action(t);
   } else if (id >= kToggleListStart && id < kUserFunctionStart) {
      TToggle *t = (TToggle *) ud;
      if (t->GetState() == 0)
         t->SetState(1);
   } else {
      TClassMenuItem *mi = (TClassMenuItem*)ud;
      fContextMenu->Action(mi);
   }
}
