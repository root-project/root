#ifndef __CINT__
#  include <qfiledialog.h> 
#  include <qstring.h> 
#endif
TString QtFileDialog() {
  // This is a small AClIC wrapper to use Qt 3.3 begin_html <a href="http://doc.trolltech.com/3.3/qfiledialog.html">QFileDialog</a> end_html class
  // See: begin_html http://doc.trolltech.com/3.3/qfiledialog.html#getOpenFileName end_html
  //
  // To use, invoke ACLiC from the ROOT prompt:
  // root [] .x QtFileDialog.C++
  //
  // To use it with no ACLiC omit the trailing "++
  // root [] .x QtFileDialog.C
  //
  // The QtFileDialog returns TString object that conatisn the selected file name.
  // returns its pointer.
  // Teh nmacro QtMultiFileDialog.C provides an adavcned example.
  //
  // The full list of the Qt classes availbe from Cint is defined by
  // begin_html <http://root.bnl.gov/QtRoot/htmldoc/src/qtclasses.h.html">by $ROOTSYS/cint/lib/qtclasses.h</a> end-html
  // Author: Valeri Fine   23/03/2006
  
#ifdef __CINT__
  // Load the qt cint dictionary.
  // One is recommended to do that at once somewhere.
  // For example  from his/her custom rootlogon.C script
  gSystem->Load("$ROOTSYS/cint/include/qtcint");
#endif   
  QString fileName = QFileDialog::getOpenFileName ();
  return TString((const char *)fileName);
}

 
