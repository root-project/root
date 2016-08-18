/// \file
/// \ingroup tutorial_gui
/// This is a small ROOT macro to use Qt 3.3 class :[QFileDialog](https://doc.qt.io/archives/3.3/qfiledialog.html)
/// See: [https://doc.qt.io/archives/3.3/qfiledialog.html#getOpenFileNames](https://doc.qt.io/archives/3.3/qfiledialog.html#getOpenFileNames)
///
/// To use, invoke ACLiC from the ROOT prompt:
/// ~~~
/// root [] .x QtMultiFileDialog.C++
/// ~~~
///
/// To use it with no ACLiC, omit the trailing "++"
/// ~~~
/// root [] .x QtMultiFileDialog.C
/// ~~~
///
/// The QtMultiFileDialog creates TList of TObjString objects and
/// returns its pointer.
///
/// The "QtFileDialog.C" macro provides the simplified version of the "QtMultiFileDialog.C"
///
/// Option: you can change the look and feel of the Qt file dialog
/// ======= by providing the optional parameter "style":
///         The number of the available styles is defined by your local
///         Qt installation.
///         Try: "windows", "motif", "kde", "platinum" etc
///
/// The full list of the Qt classes available from Cint is defined by
/// [by $ROOTSYS/cint/lib/qtclasses.h](http://root.bnl.gov/QtRoot/htmldoc/src/qtclasses.h.html)
///
/// All Qt classes can be used from ACLiC though.
///
/// \macro_code
///
/// \author Valeri Fine   23/03/2006

#  include <QApplication>
#  include <QStyle>
#  include <QFileDialog>
#  include <QStringList>
#  include <QString>
#  include "TObjString.h"
#  include "TList.h"
#  include <string>

TList *QtMultiFileDialog(const char *style="") {

   QStyle *saveStyle =  0;
   if (!QString(style).isEmpty()) {
      saveStyle = QApplication::style();
      QApplication::setStyle(style);
   }
   TList *listOfNames = new TList();
   QStringList files = QFileDialog::getOpenFileNames ();
   QStringList::Iterator it = files.begin();
   while ( it != files.end() ) {
      std::string flnm = (*it).toStdString();
      printf ("Next file selected: %s\n", flnm.c_str() );
      // Convert QString to TObjString and add it to the output
      listOfNames->Add(new TObjString(flnm.c_str()));
      ++it;
   }
   // Restore the style
   if (saveStyle) QApplication::setStyle(saveStyle);
   printf ("\nThe TList of the file names contains:");
   printf ("\n-------------------------------------\n");
   listOfNames->ls();
   return listOfNames;
}
