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
        }
	RootCaloHit(float e, float t, int i, 
                    const std::string& s, unsigned int id) : energy (e), time (t), itra (i), mycell(s, id), mynocell(7)
           {
             mycellnull = 0;
             mycellfix = new RootPCfix(4) ;
             mycellvirt = new RootPCvirt(5);
             mynocellp = new RootPCnodict(8);
           }
	virtual ~RootCaloHit() {}

        void myPrint() {
          //return;
          Dump();
          mycell.Print();
          mynocell.Print();
          if (mycellfix) mycellfix->Print();
          mycellvirt->Print();
          mynocellp->Print();
          
        }
protected:
	float energy; 
	float time; 
	int itra; 
public:
	RootPCellID mycell; 
        RootPCnodict mynocell;
	RootPCellID *mycellnull; 
	RootPCellID *mycellfix; //
	RootPCellID *mycellvirt; //
	RootPCellID *mynocellp; //
	ClassDef(RootCaloHit,1)
}; 

#endif /* !defined(_ROOT_CALO_HIT_H) */
