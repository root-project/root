#include "TFile.h"
#include "TClonesArray.h"
#include "RootCaloHit.h"
#include "RootData.h"
#include "TROOT.h"
#include "TClass.h"
#include "TStreamerInfo.h"

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
int main(int argc, char **argv)
#endif
{
   gDebug = 0;

   bool readold = false;

#ifndef SHARED
   if (argc==2) {
      TString arg2(argv[1]);
      readold = (arg2 == "-readold");
   }
#endif

   TString outputname("mytest.root");
   TString inputname;
   if (readold) inputname = "oldtest.root";
   else inputname = outputname;

   TFile f(outputname,"RECREATE");


    RootData rsrd((char *)"RootCaloHit", (unsigned)10);
    new(rsrd.data[0]) RootCaloHit(1.1,1.2,2,"test01",3);
    RootCaloHit * ph = ((RootCaloHit*)rsrd.data[0]);

#if 1
    std::cerr << "Writing Hit" << std::endl;
    ph->Write("myhit");
    ph->myPrint();
    
    //std::cerr << "Automatic Print" << std::endl;
    //ph->Dump();
    std::cerr << std::endl;
    
   TFile *input = 0;
   if (readold) {
      input = new TFile(inputname);
      if (input->IsZombie()) return 1;
   } else input = &f;
   f.cd();

    std::cerr << "Reading Hit" << std::endl;
    ph = (RootCaloHit*)input->Get("myhit");
    ph->myPrint();
    std::cerr << std::endl;
#endif

#if 1
    std::cerr << "Writing Data" << std::endl;
    rsrd.Write("mydata");
    ((RootCaloHit*)rsrd.data[0])->myPrint();
    std::cerr << std::endl;
    
    std::cerr << "Reading Data" << std::endl;
    RootData *p = (RootData*)input->Get("mydata");
    ((RootCaloHit*)p->data[0])->myPrint();
    std::cerr << std::endl;
#endif

    
    RootPCobject *myobjp = new RootPCobject(110);
    RootPCobject2*myobjp2 = new RootPCobject2(111);

    std::cerr << "Writing multiple inherited objects" << std::endl;
    myobjp->Print();
    myobjp->Write("obj1");
    myobjp2->Print();
    myobjp2->Write("obj2");

    std::cerr << "Reading multiple inherited objects with old methods" << std::endl;
    myobjp = (RootPCobject*)((void*)input->Get("obj1"));
    //std::cerr << typeid(*myobjp).name() << endl;
    //myobjp->Print();
    
    myobjp2 = (RootPCobject2*)input->Get("obj2");
    //std::cerr << typeid(*myobjp2).name() << endl;
    myobjp2->Print();

    std::cerr << "Reading multiple inherited objects with new methods" << std::endl;
    myobjp = dynamic_cast<RootPCobject *>(input->Get("obj1"));
    myobjp->Print();

    myobjp2 = dynamic_cast<RootPCobject2 *>(input->Get("obj2"));
    myobjp2->Print();

    std::cerr << "Reading multiple inherited objects with newer methods" << std::endl;
    //myobjp = (MyClass*)input->Get("obj1",RootPCobject::Class());

    std::cerr << "Check the base class offset in the case of private (and protected) inheritance" << std::endl;

    TClass *cl = gROOT->GetClass(typeid(RootPrivPCobject));
    cl->Print();
    std::cerr << "base class offset for RootPrivPCobject's TObject is: " << 
       cl->GetBaseClassOffset(TObject::Class()) << std::endl;
    //RootPrivPCobject *ppo = new RootPrivPCobject;
    //std::cerr << "When it should be " << ((char*)(TObject*)ppo)-((char*)ppo) << std::endl;
#ifdef INC_FAILURE
    std::cerr << "base class offset for RootPrivPCobject's RootPCellID is: " << 
       cl->GetBaseClassOffset(gROOT->GetClass(typeid(RootPCellID))) << std::endl;
#endif

    cl = gROOT->GetClass(typeid(RootPrivPCobject2));
    cl->Print();
    std::cerr << "base class offset for RootPrivPCobject2's TObject is: " << 
       cl->GetBaseClassOffset(TObject::Class()) << std::endl;
#ifdef INC_FAILURE
    std::cerr << "base class offset for RootPrivPCobject2's RootPCellID is: " << 
       cl->GetBaseClassOffset(gROOT->GetClass(typeid(RootPCellID))) << std::endl;
#endif

    f.Close();
    if (input!=&f) delete input;
        
    // Verify the class version:
    bool result = true;
    result &= verifyVersion("RootPCvirt",-1);
    //no header file version setting yet result &= verifyVersion("RootPCellID",2);
    result &= verifyVersion("RootPCfix",3);
    result &= verifyVersion("RootPCnodict",1);
    //no class template wide version setting yet result &= verifyVersion("RootPCtemp<int>",5);
    
#ifdef INC_FAILURE
    RootCaloHit::Class()->GetStreamerInfo()->ls();
#endif
    
    return !result;
}

ClassImp(RootCaloHit)
ClassImp(RootData)
ClassImp(RootPCellID)
