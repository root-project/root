#include "TFile.h"

void execTClassAtTearDown()
{
   TFile f("empty.root", "READ");

   // Because of this call to exit, even-though the TFile is
   // on the stack, it will not be destructed.
   // Furthermore, TROOT::CloseFile will not destructed either
   // since it is on the stack.
   // Subsequently the TFile object will still be on the list of
   // files when it is being deleted and thus will need to
   // interogate the object *AND* its TClass.
   // We setup this exmaple so that at the time `exit` is called
   // the `TClass` for `TFile` has not been created yet.
   // Consequently, its creation will be asked for during
   // the deletion of the list of files ... which is in the
   // middle of the tear down ... if a TClass object is created
   // at that time, it will use some resources (static tables)
   // that have already been tear down, resulting in crashes.
   // The new behavior is to sort-of ignore this late request
   // and build only a dummy TClass object (which does not
   // do any registrations).
   exit(0);
}
