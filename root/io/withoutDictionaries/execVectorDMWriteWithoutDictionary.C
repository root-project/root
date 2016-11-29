#include "TFile.h"

void execVectorDMWriteWithoutDictionary()
{
   TFile f("VectorWithoutDictionary.root", "recreate");
   ECont  obj;
   obj.elems.push_back( Elem(5) );
   f.WriteObject(&obj, "myobj");
   f.Close();
}


