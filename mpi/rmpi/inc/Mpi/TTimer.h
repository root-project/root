// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TTimer
#define ROOT_Mpi_TTimer

#include<Mpi/Globals.h>

namespace ROOT {
   namespace Mpi {
      class TTimer: public TObject {
         Double_t fStarTime;
         MPI_Comm  fComm;
      public:
         TTimer() {};
         void Start();
         void ReStart();
         Double_t GetElapsed() const;
         static void Sleep(Double_t msec);
         Bool_t IsGlobal() const;
         void Print() const;
         ClassDef(TTimer, 0)
      };
   }//end namespace Mpi
}//end namespace ROOT


#endif
