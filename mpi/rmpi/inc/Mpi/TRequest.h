// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TRequest
#define ROOT_Mpi_TRequest

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include<functional>
#include<TObject.h>
#include<mpi.h>

namespace ROOT {
   namespace Mpi {
      class TStatus;
      class TRequest: public TObject {
         friend class TCommunicator;
         friend class TGrequest;
         friend class TPrequest;
      protected:
         std::function<void(void)> fUnserialize; //function to unserialize object at wait or test
         MPI_Request fRequest;
      public:
         TRequest();
         TRequest(const TRequest &obj);
         virtual ~TRequest() {}
         TRequest(MPI_Request r);

         TRequest &operator=(const TRequest &r);

         Bool_t operator== (const TRequest &a);
         Bool_t operator!= (const TRequest &a);

         TRequest &operator= (const MPI_Request &i);

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

         ClassDef(TRequest, 1)
      };


      class TPrequest: public TRequest {
      public:
         TPrequest() { }
         TPrequest(const TRequest &p) : TRequest(p) { }
         TPrequest(const MPI_Request &i) : TRequest(i) { }
         virtual ~TPrequest() { }

         TPrequest &operator=(const TRequest &r)
         {
            fRequest = r.fRequest;
            return *this;
         }
         TPrequest &operator=(const TPrequest &r)
         {
            fRequest = r.fRequest;
            return *this;
         }

         virtual void Start();
         static void Startall(int count, TPrequest array_of_requests[]);

         ClassDef(TPrequest, 2)
      };

      class TGrequest: public TRequest {
      public:
         typedef Int_t Query_function(void *, TStatus &);
         typedef Int_t Free_function(void *);
         typedef Int_t Cancel_function(void *, Bool_t);

         TGrequest() {}
         TGrequest(const TRequest &req) : TRequest(req) {}
         TGrequest(const MPI_Request &req) : TRequest(req) {}
         virtual ~TGrequest() {}

         TGrequest &operator=(const TRequest &req)
         {
            fRequest = req.fRequest;
            return (*this);
         }

         TGrequest &operator=(const MPI_Request &req)
         {
            fRequest = req;
            return (*this);
         }

         TGrequest &operator=(const TGrequest &req)
         {
            fRequest = req.fRequest;
            return (*this);
         }

         static TGrequest Start(Int_t(*Query_fn)(void *, TStatus &), Int_t(*Free_fn)(void *), Int_t(*Cancel_fn)(void *, Bool_t), void *);

         virtual void Complete();


         // Type used for intercepting Generalized requests in the C++ layer so
         // that the type can be converted to C++ types before invoking the
         // user-specified C++ callbacks.
         //
         struct Intercept_data_t {
            void *id_extra;
            TGrequest::Query_function *id_cxx_query_fn;
            TGrequest::Free_function *id_cxx_free_fn;
            TGrequest::Cancel_function *id_cxx_cancel_fn;
         };
         ClassDef(TGrequest, 2)
      };



   }//end namespace Mpi
}//end namespace ROOT


#endif
