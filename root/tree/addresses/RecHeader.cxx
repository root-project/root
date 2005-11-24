#include <iostream>
using namespace std;

#include "RecHeader.h"


ClassImp(RecHeader)

// Instantiate associated template Record classes
// **********************************************
#include "RecRecordImp.cxx"
template class RecRecordImp<RecHeader>;

#include "RecDataRecord.cxx"
template class RecDataRecord<RecHeader>;


// Definition of methods (alphabetical order)
// ***************************************************


void RecHeader::Print(Option_t* /* option */) const {
  //
  //  Purpose:  Print header in form supported by TObject::Print.
  //
  //  Arguments: option (not used)
  //
  //  Return:  none.
  //
  //  Contact:   S. Kasahara
  // 

  cout << "RecHeader::Print " << endl;
  fContext.Print();
  return;

}







