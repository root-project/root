#include "TClonesArray.h"
#include "RtObj.h"
#include "TFile.h"
#include <iostream>

using namespace std;

int main()
 {
  // begin
   //  TROOT rdummy("dummy","dummy") ;
   //gDebug =1;
  // write
  TFile * file =  new TFile("RtObj.root","recreate") ;
  TClonesArray array("RtObj<TNamed>",2) ;
  new (array[0]) RtObj<TNamed>(TNamed("hello","")) ;
  new (array[1]) RtObj<TNamed>(TNamed(" world","")) ;
  cout
    <<"write: "
    <<((RtObj<TNamed> *)(array[0]))->GetName()
    <<((RtObj<TNamed> *)(array[1]))->GetName()
    <<endl ;
  array.Write("clones",TObject::kSingleKey) ;
  file->Close() ;
  delete file ;

  // read
  file =  new TFile("RtObj.root","read") ;
  TClonesArray * parray = (TClonesArray *)(file->Get("clones")) ;
  cout
    <<"read: "
    <<((RtObj<TNamed> *)((*parray)[0]))->GetName()
    <<((RtObj<TNamed> *)((*parray)[1]))->GetName()
    <<endl ;
  file->Close() ;
  delete file ;

  // end
  return 0 ;
 }



