#include <TH1D.h>
#include <TRef.h>
#include <TProcessID.h>
#include <TMath.h>

bool CheckRef(TRef &ref, TObject *expected, const char *label)
{
   Printf("%s referenced object: %p vs. expected %p",label,ref.GetObject(),expected);

   if (ref.GetObject() == nullptr) {
      Fatal("CheckRef","The ref '%s' could not find its referencee: %p %s\n",label,expected,expected->GetName());
      return false;
   }
   if (ref.GetObject() != expected) {
      Fatal("CheckRef","The ref '%s' found a different object than expected: %p %s vs %p %s\n",label,ref.GetObject(),ref.GetObject()->GetName(),expected,expected->GetName());
      return false;
   }
   return true;
}


void manyRefTest(){
   Printf("Session process ID: %p",TProcessID::GetSessionProcessID());
   auto originalProcessID = TProcessID::GetSessionProcessID();

   TH1D *h1=new TH1D();
   TRef r1=h1;
   Printf("First reference created, pointing to object %p", h1);
   Printf("Process ID %p",TProcessID::GetPID());
   Printf("Check object counted: %d",TProcessID::GetObjectCount());
   Printf("\n \n Now incrementing number of references \n\n");

   // Speed up the example.
   UInt_t startPoint = TMath::Power(2,24)-20;
   TProcessID::SetObjectCount(startPoint+1);

   Printf("Check object counted after skip: %d",TProcessID::GetObjectCount());

   for(Int_t k=startPoint;k<TMath::Power(2,24)-2;k++){// just increment number of TRef used
      TH1D *htemp=new TH1D();TRef rtemp=htemp; delete htemp;
   }


   Printf("Check Obj count after creations: %d,",TProcessID::GetObjectCount());
   Printf("and Process ID %p",TProcessID::GetPID());

   auto secondProcessID = TProcessID::GetPID();
   if (secondProcessID != originalProcessID) {
      Fatal("manyRefTest","SessionID switched sooner than expected (%p vs %p and %d ref counts)",originalProcessID,secondProcessID,TProcessID::GetObjectCount());
   }

   Printf("Now passing limit of TRef per ProcessID");

   TH1D *h2 = new TH1D();
   TRef r2(h2);
   Printf("Obj count: %d,",TProcessID::GetObjectCount());
   Printf("Process ID %p",TProcessID::GetPID());

   Printf("r2 process ID: %p",r2.GetPID());

   Printf("r2 referenced object: %p vs. expected %p",r2.GetObject(),h2);
   Printf("r referenced object (first reference): %p vs. expected %p",r1.GetObject(),h1);
   Printf("Check session process ID: %p",TProcessID::GetSessionProcessID());

   CheckRef(r1,h1,"r1");
   CheckRef(r2,h2,"r2");

   auto thirdProcessID = TProcessID::GetPID();
   if (thirdProcessID == originalProcessID) {
      Fatal("manyRefTest","SessionID switched later than expected (%p vs %p and %d ref counts)",originalProcessID,thirdProcessID,TProcessID::GetObjectCount());
   }

   Printf("Try another TRef");
   TH1D *h3 = new TH1D();
   TRef r3(h3);
   Printf("r3 process ID: %p",r3.GetPID());
   Printf("r3 referenced object: %p vs. expected %p",r3.GetObject(),h3);

   CheckRef(r3,h3,"r3");

   Printf("Try another TRef");
   TH1D *h4 = new TH1D();
   TRef r4 = h4;
   Printf("r4 process ID: %p",r4.GetPID());
   Printf("r4 referenced object: %p vs. expected %p",r4.GetObject(),h4);

   CheckRef(r4,h4,"r4");

}

void oneMore()
{

   Printf("Try another TRef");
   TH1D *h3=new TH1D();
   TRef r3=h3;
   Printf("r3 process ID: %p",r3.GetPID());
   Printf("r3 referenced object: %p vs. expected %p",r3.GetObject(),h3);
   
}

void assertManyRefs() {
   manyRefTest();
}
