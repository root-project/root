#include "TFile.h"
#include "TClonesArray.h"
#include "RootCaloHit.h"
#include "RootData.h"
#include "TROOT.h"
#include "TClass.h"

RootClassVersion(RootPCfix,3)


bool verifyVersion(const char* name, Short_t vers) {
   TClass * cl = gROOT->GetClass(name);
   if (cl->GetClassVersion() != vers ) {
      std::cerr << "The version of " << name << " is " 
                << cl->GetClassVersion() << " instead of " 
                << vers << std::endl;
      return false;
   }
   return true;
}


#ifdef SHARED
int maintest()
#else
int main()
#endif
{
    gDebug = 0;
	TFile f("mytest.root","RECREATE");
	RootData rsrd((char *)"RootCaloHit", (unsigned)10);
	new(rsrd.data[0]) RootCaloHit(1.1,1.2,2,"test01",3);
    RootCaloHit * ph = ((RootCaloHit*)rsrd.data[0]);

// On linux this prints 0 indicating that the object is not on the heap!
//std::cerr << "fBits " << ph->TestBit(0x01000000) << " for " << ph << std::endl;

    std::cerr << "Writing Hit" << std::endl;
    ph->Write("myhit");
    ph->myPrint();

        //std::cerr << "Automatic Print" << std::endl;
        //ph->Dump();
        std::cerr << std::endl;

        std::cerr << "Reading Hit" << std::endl;
        ph = (RootCaloHit*)f.Get("myhit");
        ph->myPrint();
        std::cerr << std::endl;

#if 1
        std::cerr << "Writing Data" << std::endl;
	rsrd.Write("mydata");
        ((RootCaloHit*)rsrd.data[0])->myPrint();
        std::cerr << std::endl;

        std::cerr << "Reading Data" << std::endl;
        RootData *p = (RootData*)f.Get("mydata");
        ((RootCaloHit*)p->data[0])->myPrint();
        std::cerr << std::endl;
#endif
	f.Close();

        // Verify the class version:
        bool result = true;
        result &= verifyVersion("RootPCvirt",1);
        //no header file version setting yet result &= verifyVersion("RootPCellID",2);
        result &= verifyVersion("RootPCfix",3);
        result &= verifyVersion("RootPCnodict",1);
        //no class template wide version setting yet result &= verifyVersion("RootPCtemp<int>",5);

        return !result;
}

ClassImp(RootCaloHit)
ClassImp(RootData)
ClassImp(RootPCellID)
