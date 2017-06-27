// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TMpiTimer
#define ROOT_Mpi_TMpiTimer

#include<Mpi/TIntraCommunicator.h>
#include<TH1F.h>

namespace ROOT {
   namespace Mpi {
   /**
    * \class TMpiTimer
    * class to measure times in the mpi environment.
    * \ingroup Mpi
    */

   class TMpiTimer : public TObject {
     Double_t fStarTime;
     TIntraCommunicator fComm;

   public:
     TMpiTimer(MPI_Comm comm);
     void Start();
     void ReStart();
     Double_t GetElapsed() const;
     static Double_t GetTick();
     static void Sleep(Double_t msec);
     Bool_t IsGlobal() const;
     void Print() const;
     TH1F *GetElapsedHist(Int_t root) const;
     ClassDef(TMpiTimer, 0)
      };
   }//end namespace Mpi
}//end namespace ROOT


#endif
