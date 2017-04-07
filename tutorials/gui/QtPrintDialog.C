/// \file
/// \ingroup tutorial_gui
/// This is a small ROOT macro to use Qt 3.3 class: [QPrinter](https://doc.qt.io/archives/3.3/qprinter.html)
/// to setup the printer via Qt "setup printer dialog"
/// See: [Printer setup dialog box](https://doc.qt.io/archives/3.3/qprinter.html#setup)
/// and print out the ROOT TCanvas object either via the "can" pointer provided or the current one.
///
/// To use, invoke ACLiC from the ROOT prompt:
/// ~~~
/// root [] .x QtPrintDialog.C++
/// ~~~
///
/// To use it with no ACLiC, omit the trailing "++"
/// ~~~
/// root [] .x QtPrintDialog.C
/// ~~~
///
/// The full list of the Qt classes available from Cint is defined by
/// [by $ROOTSYS/cint/lib/qtclasses.h](http://root.bnl.gov/QtRoot/htmldoc/src/qtclasses.h.html)
///
/// All Qt classes can be used from ACLiC though.
///
/// \macro_code
///
/// \author Valeri Fine   23/03/2006


#  include <QPrintDialog>
#  include <QPrinter>
#  include <QPainter>
#  include <QPixmap>
#  include <TCanvas.h>
#  include <TGQt.h>

void  QtPrintDialog(TVirtualPad *pad = 0) {

   TVirtualPad *pd = pad;
   if (!pd) pd = TPad::Pad(); // ->GetCanvas();
   if (pd) {
      QPrinter p;
      // Open the Qt "Setup Printer" dialog to configure the "QPrinter p" object
      QPrintDialog printDialog(&p);
      if (printDialog.exec() == QDialog::Accepted) {
         Int_t id = pd->GetPixmapID();
         QPixmap *pix = (QPixmap *)(TGQt::iwid(id));
         QPainter pnt(&p);
         pnt.drawPixmap(0,0,*pix);
      }
   } else {
      printf(" No TCanvas has been selected yet! \n");
   }
}

