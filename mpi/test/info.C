#include <TMpi.h>
using namespace ROOT::Mpi;

void info()
{
   TEnvironment env;                    // environment to start communication system
   TIntraCommunicator comm(COMM_WORLD); // Communicator to send/recv messages

   TInfo info;

   ROOT_MPI_ASSERT(info.IsEmpty());

   ROOT_MPI_ASSERT(info.HasKey("anykey") == kFALSE);

   info["int"] = 1;
   info["float"] = 0.1;
   info["double"] = 1.1;
   info["string"] = "hello";
   ROOT_MPI_ASSERT(info.GetNKeys() == 4);
   ROOT_MPI_ASSERT(info.HasKey("string") == kTRUE);
   info.Print();

   Int_t ivalue = info["int"];
   Float_t fvalue = info["float"];
   Double_t dvalue = info["double"];
   TString str = info["string"];
   ROOT_MPI_ASSERT(ivalue == (Int_t)1);
   ROOT_MPI_ASSERT(fvalue == (Float_t)0.1);
   ROOT_MPI_ASSERT(dvalue == (Double_t)1.1);
   ROOT_MPI_ASSERT(str == "hello");

   info.Delete("string");
   ROOT_MPI_ASSERT(info.GetNKeys() == 3);
   info.Set("string", "hostname");

   // duplicated info
   TInfo dupinfo = info.Dup();

   ROOT_MPI_ASSERT(dupinfo.GetNKeys() == 4);
   ivalue = dupinfo["int"];
   fvalue = dupinfo["float"];
   dvalue = dupinfo["double"];
   str = dupinfo["string"].GetValue<TString>();
   ROOT_MPI_ASSERT(ivalue == (Int_t)1);
   ROOT_MPI_ASSERT(fvalue == (Float_t)0.1);
   ROOT_MPI_ASSERT(dvalue == (Double_t)1.1);
   ROOT_MPI_ASSERT(str == "hostname");

   TInfo info1, info2;
   // testing overloaded operators == and !=
   ROOT_MPI_ASSERT(info1 == info2); // both empty the true
   info1["test1"] = "1";
   info1["test2"] = "2";
   info1["test3"] = "3";

   ROOT_MPI_ASSERT(info1 != info2);
   info2["test1"] = "1";
   info2["test2"] = "2";
   info2["test3"] = "3";

   ROOT_MPI_ASSERT(info1 == info2);
   info2["test4"] = "4";
   ROOT_MPI_ASSERT(info1 != info2);

   TInfo ninfo(INFO_NULL);
   ROOT_MPI_ASSERT(ninfo != info1);
   ROOT_MPI_ASSERT(ninfo.IsNull() == kTRUE);
   ROOT_MPI_ASSERT(ninfo.IsEmpty() == kTRUE);
}
