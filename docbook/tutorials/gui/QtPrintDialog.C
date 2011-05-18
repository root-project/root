// @(#)root/qt:$Name:  $:$Id: QtPrintDialog.C,v 1.4 2008/09/28 02:22:23 fine Exp $
// Author: Valeri Fine   23/03/2006
#ifndef __CINT__
#  include <QPrintDialog>
#  include <QPrinter> 
#  include <QPainter> 
#  include <QPixmap> 
#  include <TCanvas.h> 
#  include <TGQt.h> 
#endif
void  QtPrintDialog(TVirtualPad *pad = 0) {
  //
  // This is a small ROOT macro to use Qt 3.3 class: begin_html <a href="http://doc.trolltech.com/3.3/qprinter.html">QPrinter</a> end_html
  // to setup the printer via Qt "setup printer dialog"
  // See: begin_html <a href="http://doc.trolltech.com/3.3/qprinter.html#setup">Printer setup dialog box</a>  end_html
  // and print out the ROOT TCanvas object either via the "can" pointer provided or the current one.
  //
  // To use, invoke ACLiC from the ROOT prompt:
  // root [] .x QtPrintDialog.C++
  //
  // To use it with no ACLiC, omit the trailing "++"
  // root [] .x QtPrintDialog.C
  //
  //
  // The full list of the Qt classes available from Cint is defined by
  // begin_html <a href="http://root.bnl.gov/QtRoot/htmldoc/src/qtclasses.h.html">by $ROOTSYS/cint/lib/qtclasses.h</a> end_html
  //
  // All Qt classes can be used from ACLiC though.
  
#ifdef __CINT__
  // Load the qt cint dictionary.
  // One is recommended to do that at once somewhere.
  // For example  from his/her custom rootlogon.C script
  gSystem->Load("$ROOTSYS/cint/cint/include/qtcint");
#endif
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

