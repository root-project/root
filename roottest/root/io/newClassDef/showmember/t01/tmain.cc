#include "TFile.h"
#include "TClonesArray.h"
#include "RootCaloHit.h"
#include "RootData.h"

#include <typeinfo>

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
        
        typeid(ph);
        std::cout << &typeid(ph) << " which is " << typeid(ph).name() << std::endl; 
        std::cout << &typeid(*ph) << " which is " << typeid(*ph).name() << std::endl; 
        std::cout << &typeid(ph->mycell) << " which is " << typeid(ph->mycell).name() << std::endl; 
        std::cout << &typeid(ph->mycellvirt) << " which is " << typeid(ph->mycellvirt).name() << std::endl; 
        std::cout << &typeid(ph->mycellfix) << " which is " << typeid(ph->mycellfix).name() << std::endl; 

        std::cout << &typeid(*ph->mycellvirt) << " which is " << typeid(*ph->mycellvirt).name() << std::endl; 
        std::cout << &typeid(*ph->mycellfix) << " which is " << typeid(*ph->mycellfix).name() << std::endl; 

        std::cout << &typeid(RootPCfix) << " which is " << typeid(RootPCfix).name() << std::endl; 

#if 0
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

        std::cerr << "Writing Data" << std::endl;
	rsrd.Write("mydata");
        ((RootCaloHit*)rsrd.data[0])->myPrint();
        std::cerr << std::endl;

        std::cerr << "Reading Data" << std::endl;
        RootData *p = (RootData*)f.Get("mydata");
        ((RootCaloHit*)p->data[0])->myPrint();
        std::cerr << std::endl;

	f.Close();
#endif
        return 0;
}

ClassImp(RootCaloHit)
ClassImp(RootData)
ClassImp(RootPCellID)
