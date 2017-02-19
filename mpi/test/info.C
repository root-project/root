#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

void info()
{
   TEnvironment env;          //environment to start communication system
   TIntraCommunicator comm(COMM_WORLD);   //Communicator to send/recv messages


   TInfo info;

   assert(info.IsEmpty());

   assert(info.HasKey("anykey") == kFALSE);

   info["int"] = 1;
   info["float"] = 0.1;
   info["double"] = 1.1;
   info["string"] = "hello";
   assert(info.GetNKeys() == 4);
   assert(info.HasKey("string") == kTRUE);
   info.Print();

   Int_t ivalue = info["int"];
   Float_t fvalue = info["float"];
   Double_t dvalue = info["double"];
   TString str = info["string"];
   assert(ivalue == (Int_t)1);
   assert(fvalue == (Float_t)0.1);
   assert(dvalue == (Double_t)1.1);
   assert(str == "hello");

   info.Delete("string");
   assert(info.GetNKeys() == 3);
   info.Set("string", "hostname");

   //duplicated info
   TInfo dupinfo = info.Dup();

   assert(dupinfo.GetNKeys() == 4);
   ivalue = dupinfo["int"];
   fvalue = dupinfo["float"];
   dvalue = dupinfo["double"];
   str = dupinfo["string"].GetValue<TString>();
   assert(ivalue == (Int_t)1);
   assert(fvalue == (Float_t)0.1);
   assert(dvalue == (Double_t)1.1);
   assert(str == "hostname");

   TInfo info1, info2;
   //testing overloaded operators == and !=
   assert(info1 == info2); //both empty the true
   info1["test1"] = "1";
   info1["test2"] = "2";
   info1["test3"] = "3";

   assert(info1 != info2);
   info2["test1"] = "1";
   info2["test2"] = "2";
   info2["test3"] = "3";

   assert(info1 == info2);
   info2["test4"] = "4";
   assert(info1 != info2);

   TInfo ninfo(INFO_NULL);
   assert(ninfo != info1);
   assert(ninfo.IsNull() == kTRUE);
   assert(ninfo.IsEmpty() == kTRUE);

}

