/* C++ header file: Objectivity/DB DDL version 6.0 Oct 2000 */

#ifndef _ROOT_CALO_HIT_H
#define _ROOT_CALO_HIT_H

#include "RootPCellID.h"
#include "TObject.h"


class RootCaloHit : public TObject { 
public:
	RootCaloHit() {
          mycellfix=0;
          mycellvirt=0;
          mycellnull=0;
          for(int i=0;i<4;i++) myArrFix[i]=0;
          myArrVar=0;
          myobjp = myobj2p = 0;
        }
	RootCaloHit(float e, float t, int val, 
                    const std::string& s, unsigned int id) : 
           energy (e), time (t), itra (val), 
           mycell(s, id), 
           index(0),
           myArrVar(0),
           mynocell(7),
           myobj(8),
           myobj2(9)
           {
             mycellnull = 0;
             mycellfix = new RootPCfix(4) ;
             mycellvirt = new RootPCvirt(5);
             mynocellp = new RootPCnodict(8);
             int i = 0;
             for(i=0; i<3; i++) {
                RootPCellID id(s,id+1+i);
                myArr[i] = id;
             }
             for(i=0; i<4; i++) {
                myArrFix[i] = new RootPCtemp<int>(i);
             }

             index = 2;
             myArrVar = new RootPCellID*[index];
             myArrVar[0] = new RootPCvirt(0);
             myArrVar[1] = new RootPCfix(1);

             myobjp = new RootPCobject(10);
             myobj2p = new RootPCobject2(11);
             TObject * p1 = (RootPCobject2*)myobj2p;
             void * v1 = (void*)dynamic_cast<TObject* >(myobj2p);
             fprintf(stderr,"myobj2p %p tobj %p void %p\n",
                     myobj2p,p1,v1);
           }
	virtual ~RootCaloHit() {
           int i = 0;
           for(i = 0; i<index; i++) delete myArrFix[i];
           for(i = 0; i<index; i++) delete myArrVar[i];
           delete []myArrVar;
           delete myobjp; myobjp = 0;
             TObject * p1 = (RootPCobject2*)myobj2p;
             void * v1 = (void*)dynamic_cast<TObject* >(myobj2p);
             fprintf(stderr,"myobj2p %p tobj %p void %p\n",
                     myobj2p,p1,v1);
           delete myobj2p; myobj2p = 0;
        }

        void myPrint() {
          //return;
          Dump();
          mycell.Print();
          mynocell.Print();
          if (mycellfix) mycellfix->Print();
          mycellvirt->Print();
          mynocellp->Print();
          int i = 0;
          for(i=0; i<3; i++) myArr[i].Print();
          for(i=0; i<4; i++) if (myArrFix[i]) myArrFix[i]->Print();
          if (myArrVar) for(i=0; i<index; i++) if (myArrVar[i]) myArrVar[i]->Print();
          myobj.Print();
          myobj2.Print();
          if (myobjp) myobjp->Print();
          if (myobj2p) myobj2p->Print();
          
        }
protected:
	float energy; 
	float time; 
	int itra; 
public:
	RootPCellID      mycell;
        RootPCellID      myArr[3];
        RootPCtemp<int> *myArrFix[4]; 
        int index;
        RootPCellID    **myArrVar;    //! WAITING on Vicktor's implementation [index]
        RootPCnodict     mynocell;
	RootPCellID     *mycellnull; 
	RootPCellID     *mycellfix; //
	RootPCellID     *mycellvirt; //
	RootPCellID     *mynocellp; //

	RootPCobject     myobj;  //
	RootPCobject2    myobj2;
	RootPCellID     *myobjp; //
	RootPCellID     *myobj2p;

	ClassDef(RootCaloHit,1);
}; 

#endif /* !defined(_ROOT_CALO_HIT_H) */
