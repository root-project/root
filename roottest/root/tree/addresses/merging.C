#include "Riostream.h"

#include "TMessage.h"
#include "TTree.h"
#include "TROOT.h"

#ifndef merging_C
#define merging_C

using namespace std;

int merging() 
{
   Long_t NUMBER_OF_ENTRIES = 100;
   
   TTree* newResult = new TTree("xxx", "Argument");
   static Double_t x, y;
   newResult->Branch("x", &x, "x/D");
   newResult->Branch("y", &y, "y/D");
   for(Long_t i=0; i<NUMBER_OF_ENTRIES; ++i)
   {
      x = i;
      y = i*i;
      //fprintf(stderr,"res %lf %lf %d\n",x,y,i<NUMBER_OF_ENTRIES);
      newResult->Fill();
   }// end of for
   //  newResult->Scan("x:y");
   
   // ======================================
   
   TMessage message(kMESS_OBJECT);
   
   message.Reset();
   message.SetWriteMode();
   message.WriteObject(newResult);
   
   message.Reset();
   message.SetReadMode();
   TTree* readResult = 0;
   readResult = ((TTree*)message.ReadObject(message.GetClass()));
   readResult->SetName("yyy");
   
   // ======================================
   
   TTree* result = 0;
   
   result = readResult->CloneTree(0);
   result->SetName("zzz");
   result->Print();
   result->Show(19);
   readResult->Print();
   readResult->Show(29);
   
   cout<< "Result has " << result->GetEntries()<< " entries." << endl;
   
   TList newResultCollection;
   newResultCollection.SetOwner(kFALSE);
   newResultCollection.Add(readResult);
   
   cerr<<"Hello 1\n";
   
   result->Merge(&newResultCollection);
   
   cerr<<"Hello 2\n";
   
   cout<<result->GetEntries()<<endl;
   printf("result entries = %lld\n",result->GetEntries());
   
   // ======================================
   
   newResultCollection.Clear();
   delete newResult;
   delete readResult;
   
   return 0;
   
} // end of main

#endif
