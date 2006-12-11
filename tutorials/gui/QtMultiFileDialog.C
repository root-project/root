#ifndef __CINT__
#  include <qapplication.h> 
#  include <qstyle.h> 
#  include <qfiledialog.h> 
#  include <qstringlist.h> 
#  include <qstring.h> 
#  include "TObjString.h"
#  include "TList.h"
#endif
TList *QtMultiFileDialog(const char *style="") {
  // This is a small AClIC wrapper to use Qt 3.3 begin_html <a href="http://doc.trolltech.com/3.3/qfiledialog.html">QFileDialog</a> end_html class
  // See: begin_html http://doc.trolltech.com/3.3/qfiledialog.html#getOpenFileNames end_html
  //
  // To use, invoke ACLiC from the ROOT prompt:
  // root [] .x QtMultiFileDialog.C++
  //
  // To use it with no ACLiC omit the trailing "++
  // root [] .x QtMultiFileDialog.C
  //
  // The QtMultiFileDialog creates TList of TObjString objects and 
  // returns its pointer. 
  //
  // The "QtFileDialog.C" macro rpobvdies the simplified version of the "QtMultiFileDialog.C"
  //
  // Option: you can change the look and feel of the Qt file dialog
  // ======= by providing the "style" optional parameter:
  //         The number of the available styles is defined by your local
  //         Qt installation. 
  //         Try: "windows", "motif", "kde", "platinum" etc
  //
  // The full list of the Qt classes availbe from Cint is defined by
  // begin_html <http://root.bnl.gov/QtRoot/htmldoc/src/qtclasses.h.html">by $ROOTSYS/cint/lib/qtclasses.h</a> end-html
  //Author: Valeri Fine 
  
#ifdef __CINT__
  // Load the qt cint dictionary.
  // One is recommended to do that at once somewhere.
  // For example  from his/her custom rootlogon.C script
  gSystem->Load("$ROOTSYS/cint/include/qtcint");
#endif   
  QStyle *saveStyle =  0;
  if (!QString(style).isEmpty()) { 
     saveStyle = &QApplication::style();
     QApplication::setStyle(style);
  }
  TList *listOfNames = new TList();
  QStringList files = QFileDialog::getOpenFileNames ();
  QStringList::Iterator it = files.begin();
  while ( it != files.end() ) {
      printf ("Next file selected: %s\n", (const char *)(*it));
      // Convert QString to TObjString and add it to the output
      listOfNames->Add(new TObjString((const char *)(*it)));
      ++it;
  }
  // Restore the style
  if (saveStyle) QApplication::setStyle(saveStyle);
  printf ("\nThe TList of the file names contains:");
  printf ("\n-------------------------------------\n");
  listOfNames->ls();
  return listOfNames;
}

 
