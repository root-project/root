#ifndef _ROOT_CALO_HIT_H
#define _ROOT_CALO_HIT_H

#include "RootPCellID.h"
#include "TObject.h"
#include "TClass.h"


class RootCaloHit : public TObject {
public:
   RootCaloHit() : index(0),objVarArr(0)
   {
      for(int i=0;i<4;i++) myArrFix[i]=0;
      myArrVar=0;

      mycellnull=0;
      mycellfix=0;
      mycellvirt=0;
      mynocellp=0;

      myobjp = myobj2p = 0;
      myobjdp = 0;
      myobj2dp = 0;
      myobjSt = new RootPCobject(0);
      myobj2St = new RootPCobject2(0);
      mytobjp = mytobj2p = 0;
      mytobjSt = new RootPCobject(0);
      mytobj2St = new RootPCobject2(0);
   }
   RootCaloHit(float e, float t, int val,
               const std::string& s, unsigned int id) :
      energy (e), time (t), itra (val),
      mycell(s, id),
      index(0),
      myArrVar(0),
      objVarArr(0),
      mynocell(7),
      myobj(8),
      myobj2(9)
   {
      mycellnull = 0;
      mycellfix = new RootPCfix(4) ;
      mycellvirt = new RootPCvirt(5);
#ifndef __CLING__
      mynocellp = new RootPCnodict(8);
#else
      // Use -1 to be able to determine that this line is
      // really not executed in the test.
      mynocellp = (RootPCellID*)-1;
#endif
      int i = 0;
      for(i=0; i<3; i++) {
         RootPCellID cell(s,id+1+i);
         myArr[i] = cell;
      }
      for(i=0; i<4; i++) {
         myArrFix[i] = new RootPCtemp<int>(i);
      }

      index = 2;
      myArrVar = new RootPCellID*[index];
      myArrVar[0] = new RootPCvirt(0);
      myArrVar[1] = new RootPCfix(1);

      objVarArr = new RootPCellID[index];
      objVarArr[0] = RootPCellID("varr",3);
      objVarArr[1] = RootPCellID("varr",4);

      myobjp = new RootPCobject(10);
      myobj2p = new RootPCobject2(11);
      myobjSt = new RootPCobject(12);
      myobj2St = new RootPCobject2(13);

      mytobjp = new RootPCobject(14);
      mytobj2p = new RootPCobject2(15);
      mytobjSt = new RootPCobject(16);
      mytobj2St = new RootPCobject2(17);

      myobjdp = new RootPCobject(18);
      myobj2dp = new RootPCobject2(19);

      for(i=0;i<2;i++) {
         RootPCobject  r1(20+2*i);
         RootPCobject2 r2(21+2*i);
         myobjarr[i] = r1;
         myobjarr2[i]= r2;
      }

   }
   ~RootCaloHit() override
   {
      int i = 0;
      for(i = 0; i<4; i++) delete myArrFix[i];
      for(i = 0; i<index; i++) delete myArrVar[i];
      delete []myArrVar;
      delete []objVarArr;
      delete myobjp; myobjp = 0;
      delete myobj2p; myobj2p = 0;

      delete myobjp; myobjp = 0;
      delete myobj2p; myobj2p = 0;

      delete myobjdp; myobjp = 0;
      delete myobj2dp; myobj2p = 0;

      delete myobjSt; myobjSt = 0;
      delete myobj2St; myobj2St = 0;

      delete mytobjp; mytobjp = 0;
      delete mytobj2p; mytobj2p = 0;

      delete mytobjSt; mytobjSt = 0;
      delete mytobj2St; mytobj2St = 0;
   }

   void myPrint()
   {
      //return;
      IsA()->Dump(this, true /*noaddr*/);
      mycell.Print();
      mynocell.Print();
      if (mycellfix) mycellfix->Print();
      mycellvirt->Print();
      mynocellp->Print();
      int i = 0;
      for(i=0; i<3; i++) myArr[i].Print();
      for(i=0; i<4; i++) if (myArrFix[i]) myArrFix[i]->Print();
      /*           fprintf(stderr,"myArrVar is %p %p\n",myArrVar,objVarArr); */
      if (myArrVar) for(i=0; i<index; i++) if (myArrVar[i]) myArrVar[i]->Print();
      if (objVarArr) for(i=0; i<index; i++) objVarArr[i].Print();
      myobj.Print();
      myobj2.Print();

      if (myobjp) myobjp->Print();
      if (myobj2p) myobj2p->Print();

      myobjSt->Print();
      myobj2St->Print();

      if (mytobjp) dynamic_cast<RootPCobject *>(mytobjp)->Print();
      if (mytobj2p) dynamic_cast<RootPCobject2 *>(mytobj2p)->Print();
      dynamic_cast<RootPCobject *>(mytobjSt)->Print();
      dynamic_cast<RootPCobject2 *>(mytobj2St)->Print();

      if (myobjdp) myobjdp->Print();
      if (myobj2dp) myobj2dp->Print();

      for(i=0;i<2;i++) {
         myobjarr[i].Print();
         myobjarr2[i].Print();
      }

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
   RootPCellID    **myArrVar;    //![index]  WAITING on Vicktor's implementation [index]
   RootPCellID     *objVarArr;   //![index]  Not implemented yet ... will it ever?
   RootPCnoRequestedDict     mynocell;
   RootPCellID     *mycellnull;
   RootPCellID     *mycellfix; //
   RootPCellID     *mycellvirt; //
   RootPCellID     *mynocellp; //

#ifndef BROKEN_MULTI
   RootPCobject     myobj;    //
   RootPCobject2    myobj2;   //
   RootPCellID     *myobjp;   //
   RootPCellID     *myobj2p;
   RootPCobject    *myobjSt;  //->
   RootPCobject2   *myobj2St; //->
   TObject         *mytobjp;
   TObject         *mytobj2p;
   TObject         *mytobjSt; //->
   TObject         *mytobj2St;//->

   RootPCobject    *myobjdp;  //
   RootPCobject2   *myobj2dp; //

   RootPCobject     myobjarr[2];    //
   RootPCobject2    myobjarr2[2];   //
#else
   RootPCobject     myobj;          //!
   RootPCobject2    myobj2;         //!
   RootPCellID     *myobjp;         //!
   RootPCellID     *myobj2p;        //!
   RootPCobject    *myobjSt;        //!->
   RootPCobject2   *myobj2St;       //!->
   TObject         *mytobjp;        //!
   TObject         *mytobj2p;       //!
   TObject         *mytobjSt;       //!->
   TObject         *mytobj2St;      //!->

   RootPCobject    *myobjdp;         //!
   RootPCobject2   *myobj2dp;        //!

   RootPCobject     myobjarr[2];     //!
   RootPCobject2    myobjarr2[2];    //!
#endif

   ClassDefOverride(RootCaloHit,1)
};

#endif /* !defined(_ROOT_CALO_HIT_H) */

