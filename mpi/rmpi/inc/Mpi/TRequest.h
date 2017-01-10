// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TRequest
#define ROOT_Mpi_TRequest

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include<TObject.h>
#include<mpi.h>

namespace ROOT {
   namespace Mpi {
      class TStatus;
      class TRequest: public TObject {
         friend class TCommunicator;
      protected:
         MPI::Request fRequest;
      public:
         TRequest();
         TRequest(const TRequest &obj);
         virtual ~TRequest() {}
         TRequest(MPI_Request r);
         TRequest(const MPI::Request &r);

         TRequest &operator=(const TRequest &r);

         Bool_t operator== (const TRequest &a);
         Bool_t operator!= (const TRequest &a);

         TRequest &operator= (const MPI_Request &i);
         TRequest &operator= (const MPI::Request &i);

         operator MPI_Request() const
         {
            return fRequest;
         }

         virtual void Wait(TStatus &status);

         virtual void Wait();

         virtual Bool_t Test(TStatus &status);

         virtual Bool_t Test();

         virtual void Free(void);

         static Int_t WaitAny(Int_t count, TRequest array[], TStatus &status);

         static Int_t WaitAny(Int_t count, TRequest array[]);

         static Bool_t TestAny(Int_t count, TRequest array[], Int_t &index, TStatus &status);

         static Bool_t TestAny(Int_t count, TRequest array[], Int_t &index);

         static void WaitAll(Int_t count, TRequest req_array[], TStatus stat_array[]);

         static void WaitAll(Int_t count, TRequest req_array[]);

         static Bool_t TestAll(Int_t count, TRequest req_array[], TStatus stat_array[]);

         static Bool_t TestAll(Int_t count, TRequest req_array[]);

         static Int_t WaitSome(Int_t incount, TRequest req_array[], Int_t array_of_indices[], TStatus stat_array[]) ;

         static Int_t WaitSome(Int_t incount, TRequest req_array[], Int_t array_of_indices[]);

         static Int_t TestSome(Int_t incount, TRequest req_array[], Int_t array_of_indices[], TStatus stat_array[]);

         static Int_t TestSome(Int_t incount, TRequest req_array[], Int_t array_of_indices[]);

         virtual void Cancel(void) const;

         virtual Bool_t GetStatus(TStatus &status) const;

         virtual Bool_t GetStatus() const;

         operator MPI::Request() const
         {
            return fRequest;
         }

         ClassDef(TRequest, 1)
      };
   }//end namespace Mpi
}//end namespace ROOT


#endif
