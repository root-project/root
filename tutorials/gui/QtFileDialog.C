// @(#)root/qt:$Name:  $:$Id$
// Author: Valeri Fine   23/03/2006
#ifndef __CINT__
#  include <QFileDialog>
#  include <QString>
#  include "TString.h"
#  include <string>
#endif
TString QtFileDialog() {
  // This is a small ROOT macro to use Qt 3.3 class: begin_html <a href="http://doc.trolltech.com/3.3/qfiledialog.html">QFileDialog</a> end_html
  // See: begin_html <a href="http://doc.trolltech.com/3.3/qfiledialog.html#getOpenFileName">http://doc.trolltech.com/3.3/qfiledialog.html#getOpenFileName</a> end_html
  //
  // To use, invoke ACLiC from the ROOT prompt:
  // root [] .x QtFileDialog.C++
  //
  // To use it with no ACLiC, omit the trailing "++"
  // root [] .x QtFileDialog.C
  //
  // The QtFileDialog returns TString object that contains the selected file name.
  // returns its pointer.
  // The macro QtMultiFileDialog.C provides an advanced example.
  //
  // The full list of the Qt classes available from Cint is defined by
  // begin_html <a href="http://root.bnl.gov/QtRoot/htmldoc/src/qtclasses.h.html">by $ROOTSYS/cint/lib/qtclasses.h</a> end_html
  //
  // All Qt classes can be used from ACLiC though.

#ifdef __CINT__
  // Load the qt cint dictionary.
  // One is recommended to do that at once somewhere.
  // For example from one's custom rootlogon.C script
  gSystem->Load("$ROOTSYS/cint/cint/include/qtcint");
#endif
  QString fileName = QFileDialog::getOpenFileName ();
  std::string flnm = fileName.toStdString();
  return TString(flnm.c_str());
}

