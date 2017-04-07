/// \file
/// \ingroup tutorial_gui
/// This is a small ROOT macro to use Qt 3.3 class: [QFileDialog](https://doc.qt.io/archives/3.3/qfiledialog.html)
/// See: [https://doc.qt.io/archives/3.3/qfiledialog.html#getOpenFileName](https://doc.qt.io/archives/3.3/qfiledialog.html#getOpenFileName)
///
/// To use, invoke ACLiC from the ROOT prompt:
/// ~~~
/// root [] .x QtFileDialog.C++
/// ~~~
///
/// To use it with no ACLiC, omit the trailing "++"
/// ~~~
/// root [] .x QtFileDialog.C
/// ~~~
///
/// The QtFileDialog returns TString object that contains the selected file name.
/// returns its pointer.
/// The macro QtMultiFileDialog.C provides an advanced example.
///
/// The full list of the Qt classes available from Cint is defined by
/// begin_html [by $ROOTSYS/cint/lib/qtclasses.h](http://root.bnl.gov/QtRoot/htmldoc/src/qtclasses.h.html)
///
/// All Qt classes can be used from ACLiC though.
///
/// \macro_code
///
/// \author Valeri Fine   23/03/2006

#  include <QFileDialog>
#  include <QString>
#  include "TString.h"
#  include <string>


TString QtFileDialog() {
   QString fileName = QFileDialog::getOpenFileName ();
   std::string flnm = fileName.toStdString();
   return TString(flnm.c_str());
}

