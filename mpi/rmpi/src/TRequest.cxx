
#include<Mpi/TRequest.h>
#include <Mpi/TStatus.h>


using namespace ROOT::Mpi;

//______________________________________________________________________________
TRequest::TRequest(): fRequest(MPI::REQUEST_NULL) {}

//______________________________________________________________________________
TRequest::TRequest(const TRequest &obj): TObject(obj), fRequest(obj.fRequest) {}


//______________________________________________________________________________
TRequest::TRequest(MPI_Request i) : fRequest(i) { }

//______________________________________________________________________________
TRequest::TRequest(const MPI::Request &r) : fRequest(r) { }

//______________________________________________________________________________
TRequest &TRequest::operator=(const TRequest &r)
{
   fRequest = r.fRequest;
   return *this;
}

//______________________________________________________________________________
Bool_t TRequest::operator== (const TRequest &a)
{
   return (Bool_t)(fRequest == a.fRequest);
}

//______________________________________________________________________________
Bool_t TRequest::operator!= (const TRequest &a)
{
   return (Bool_t)(fRequest != a.fRequest);
}

//______________________________________________________________________________
TRequest &TRequest::operator= (const MPI_Request &i)
{
   fRequest = i;
   return *this;
}

//______________________________________________________________________________
void TRequest::Wait(TStatus &status)
{
   fRequest.Wait(status.fStatus);
}

//______________________________________________________________________________
void TRequest::Wait()
{
   fRequest.Wait();
}

//______________________________________________________________________________
Bool_t TRequest::Test(TStatus &status)
{
   return fRequest.Test(status.fStatus);
}

//______________________________________________________________________________
Bool_t TRequest::Test()
{
   return fRequest.Test();
}

//______________________________________________________________________________
void TRequest::Free(void)
{
   fRequest.Free();
}

//______________________________________________________________________________
Int_t TRequest::WaitAny(Int_t count, TRequest array[], TStatus &status)
{
   MPI::Request req_array[count];
   for (auto i = 0; i < count; i++)req_array[i] = array[i].fRequest;
   return MPI::Request::Waitany(count, req_array, status.fStatus);
}

//______________________________________________________________________________
Int_t TRequest::WaitAny(Int_t count, TRequest array[])
{
   MPI::Request req_array[count];
   for (auto i = 0; i < count; i++)req_array[i] = array[i].fRequest;
   return MPI::Request::Waitany(count, req_array);
}

//______________________________________________________________________________
Bool_t TRequest::TestAny(Int_t count, TRequest array[], Int_t &index, TStatus &status)
{
   MPI::Request req_array[count];
   for (auto i = 0; i < count; i++)req_array[i] = array[i].fRequest;
   return MPI::Request::Testany(count, req_array, index, status.fStatus);
}

//______________________________________________________________________________
Bool_t TRequest::TestAny(Int_t count, TRequest array[], Int_t &index)
{
   MPI::Request req_array[count];
   for (auto i = 0; i < count; i++)req_array[i] = array[i].fRequest;
   return MPI::Request::Testany(count, req_array, index);
}

//______________________________________________________________________________
void TRequest::WaitAll(Int_t count, TRequest array[], TStatus stat_array[])
{
   MPI::Request req_array[count];
   MPI::Status  sta_array[count];

   for (auto i = 0; i < count; i++) {
      req_array[i] = array[i].fRequest;
      sta_array[i] = stat_array[i].fStatus;
   }
   MPI::Request::Waitall(count, req_array, sta_array);
}

//______________________________________________________________________________
void TRequest::WaitAll(Int_t count, TRequest array[])
{
   MPI::Request req_array[count];
   for (auto i = 0; i < count; i++)req_array[i] = array[i].fRequest;
   MPI::Request::Waitall(count, req_array);
}

//______________________________________________________________________________
Bool_t TRequest::TestAll(Int_t count, TRequest array[], TStatus stat_array[])
{
   MPI::Request req_array[count];
   MPI::Status  sta_array[count];

   for (auto i = 0; i < count; i++) {
      req_array[i] = array[i].fRequest;
      sta_array[i] = stat_array[i].fStatus;
   }
   return MPI::Request::Testall(count, req_array, sta_array);
}

//______________________________________________________________________________
Bool_t TRequest::TestAll(Int_t count, TRequest array[])
{
   MPI::Request req_array[count];
   for (auto i = 0; i < count; i++)req_array[i] = array[i].fRequest;
   return MPI::Request::Testall(count, req_array);
}

//______________________________________________________________________________
Int_t TRequest::WaitSome(Int_t count, TRequest array[], Int_t array_of_indices[], TStatus stat_array[])
{
   MPI::Request req_array[count];
   MPI::Status  sta_array[count];

   for (auto i = 0; i < count; i++) {
      req_array[i] = array[i].fRequest;
      sta_array[i] = stat_array[i].fStatus;
   }
   return MPI::Request::Waitsome(count, req_array, array_of_indices, sta_array);
}

//______________________________________________________________________________
Int_t TRequest::WaitSome(Int_t count, TRequest array[], Int_t array_of_indices[])
{
   MPI::Request req_array[count];

   for (auto i = 0; i < count; i++) req_array[i] = array[i].fRequest;
   return MPI::Request::Waitsome(count, req_array, array_of_indices);
}

//______________________________________________________________________________
Int_t TRequest::TestSome(Int_t count, TRequest array[], Int_t array_of_indices[], TStatus stat_array[])
{
   MPI::Request req_array[count];
   MPI::Status  sta_array[count];

   for (auto i = 0; i < count; i++) {
      req_array[i] = array[i].fRequest;
      sta_array[i] = stat_array[i].fStatus;
   }
   return MPI::Request::Testsome(count, req_array, array_of_indices, sta_array);
}

//______________________________________________________________________________
Int_t TRequest::TestSome(Int_t count, TRequest array[], Int_t array_of_indices[])
{
   MPI::Request req_array[count];
   for (auto i = 0; i < count; i++) req_array[i] = array[i].fRequest;
   return MPI::Request::Testsome(count, req_array, array_of_indices);
}

//______________________________________________________________________________
void TRequest::Cancel(void) const
{
   fRequest.Cancel();
}

//______________________________________________________________________________
Bool_t TRequest::GetStatus(TStatus &status) const
{
   return fRequest.Get_status(status.fStatus);
}

//______________________________________________________________________________
Bool_t TRequest::GetStatus() const
{
   return fRequest.Get_status();
}
