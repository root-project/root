
#include<Mpi/TRequest.h>
#include <Mpi/TStatus.h>


using namespace ROOT::Mpi;

//______________________________________________________________________________
TRequest::TRequest(): fRequest(MPI_REQUEST_NULL)
{
   fUnserialize = []() {};
}

//______________________________________________________________________________
TRequest::TRequest(const TRequest &obj): TObject(obj), fRequest(obj.fRequest)
{
   fUnserialize = []() {};
}


//______________________________________________________________________________
TRequest::TRequest(MPI_Request i) : fRequest(i)
{
   fUnserialize = []() {};
}


//______________________________________________________________________________
TRequest &TRequest::operator=(const TRequest &r)
{
   fRequest = r.fRequest;
   fUnserialize = r.fUnserialize;
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
   MPI_Wait(&fRequest, &status.fStatus);
   //TODO:error control here if status is wrong
   try {
      fUnserialize();
   } catch (const std::exception &e) {
      std::cerr << "Error = " << e.what() << std::endl;
   }
}

//______________________________________________________________________________
void TRequest::Wait()
{
   MPI_Wait(&fRequest, MPI_STATUS_IGNORE);
   //TODO:error control here if status is wrong
   try {
      fUnserialize();
   } catch (const std::exception &e) {
      std::cerr << "Error = " << e.what() << std::endl;
   }
}

//______________________________________________________________________________
Bool_t TRequest::Test(TStatus &status)
{
   Int_t flag;
   MPI_Test(&fRequest, &flag, &status.fStatus);
   return (Bool_t)flag;
}

//______________________________________________________________________________
Bool_t TRequest::Test()
{
   Int_t flag;
   MPI_Test(&fRequest, &flag, MPI_STATUS_IGNORE);
   return (Bool_t)flag;
}

//______________________________________________________________________________
void TRequest::Free(void)
{
   MPI_Request_free(&fRequest);
}

//______________________________________________________________________________
Int_t TRequest::WaitAny(Int_t count, TRequest array[], TStatus &status)
{
   int index, i;
   MPI_Request *array_of_requests = new MPI_Request[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = array[i].fRequest;
   }
   MPI_Waitany(count, array_of_requests, &index, &status.fStatus);
   for (i = 0; i < count; i++) {
      array[i] = array_of_requests[i];
   }
   delete [] array_of_requests;
   return index;
}

//______________________________________________________________________________
Int_t TRequest::WaitAny(Int_t count, TRequest array[])
{
   Int_t index, i;
   MPI_Request *array_of_requests = new MPI_Request[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = array[i].fRequest;
   }
   MPI_Waitany(count, array_of_requests, &index, MPI_STATUS_IGNORE);
   for (i = 0; i < count; i++) {
      array[i] = array_of_requests[i];
   }
   delete [] array_of_requests;
   return index;
}

//______________________________________________________________________________
Bool_t TRequest::TestAny(Int_t count, TRequest array[], Int_t &index, TStatus &status)
{
   Int_t i, flag;
   MPI_Request *array_of_requests = new MPI_Request[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = array[i].fRequest;
   }
   MPI_Testany(count, array_of_requests, &index, &flag, &status.fStatus);
   for (i = 0; i < count; i++) {
      array[i] = array_of_requests[i];
   }
   delete [] array_of_requests;
   return (bool)(flag != 0 ? true : false);
}

//______________________________________________________________________________
Bool_t TRequest::TestAny(Int_t count, TRequest array[], Int_t &index)
{
   Int_t i, flag;
   MPI_Request *array_of_requests = new MPI_Request[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = array[i].fRequest;
   }
   MPI_Testany(count, array_of_requests, &index, &flag, MPI_STATUS_IGNORE);
   for (i = 0; i < count; i++) {
      array[i] = array_of_requests[i];
   }
   delete [] array_of_requests;
   return Bool_t(flag);
}

//______________________________________________________________________________
void TRequest::WaitAll(Int_t count, TRequest req_array[], TStatus stat_array[])
{
   int i;
   MPI_Request *array_of_requests = new MPI_Request[count];
   MPI_Status *array_of_statuses = new MPI_Status[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   (void)MPI_Waitall(count, array_of_requests, array_of_statuses);
   for (i = 0; i < count; i++) {
      req_array[i] = array_of_requests[i];
      stat_array[i] = array_of_statuses[i];
   }
   delete [] array_of_requests;
   delete [] array_of_statuses;
}

//______________________________________________________________________________
void TRequest::WaitAll(Int_t count, TRequest req_array[])
{
   Int_t i;
   MPI_Request *array_of_requests = new MPI_Request[count];

   for (i = 0; i < count; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   (void)MPI_Waitall(count, array_of_requests, MPI_STATUSES_IGNORE);

   for (i = 0; i < count; i++) {
      req_array[i] = array_of_requests[i];
   }

   delete [] array_of_requests;
}

//______________________________________________________________________________
Bool_t TRequest::TestAll(Int_t count, TRequest req_array[], TStatus stat_array[])
{
   Int_t i, flag;
   MPI_Request *array_of_requests = new MPI_Request[count];
   MPI_Status *array_of_statuses = new MPI_Status[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   (void)MPI_Testall(count, array_of_requests, &flag, array_of_statuses);
   for (i = 0; i < count; i++) {
      req_array[i] = array_of_requests[i];
      stat_array[i] = array_of_statuses[i];
   }
   delete [] array_of_requests;
   delete [] array_of_statuses;
   return Bool_t(flag);
}

//______________________________________________________________________________
Bool_t TRequest::TestAll(Int_t count, TRequest req_array[])
{
   Int_t i, flag;
   MPI_Request *array_of_requests = new MPI_Request[count];

   for (i = 0; i < count; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   MPI_Testall(count, array_of_requests, &flag, MPI_STATUSES_IGNORE);

   for (i = 0; i < count; i++) {
      req_array[i] = array_of_requests[i];
   }
   delete [] array_of_requests;

   return Bool_t(flag);
}

//______________________________________________________________________________
Int_t TRequest::WaitSome(Int_t incount, TRequest req_array[], Int_t array_of_indices[], TStatus stat_array[])
{
   Int_t i, outcount;
   MPI_Request *array_of_requests = new MPI_Request[incount];
   MPI_Status *array_of_statuses = new MPI_Status[incount];
   for (i = 0; i < incount; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   MPI_Waitsome(incount, array_of_requests, &outcount,
                array_of_indices, array_of_statuses);
   for (i = 0; i < incount; i++) {
      req_array[i] = array_of_requests[i];
      stat_array[i] = array_of_statuses[i];
   }
   delete [] array_of_requests;
   delete [] array_of_statuses;
   return outcount;
}

//______________________________________________________________________________
Int_t TRequest::WaitSome(Int_t incount, TRequest req_array[], Int_t array_of_indices[])
{
   Int_t i, outcount;
   MPI_Request *array_of_requests = new MPI_Request[incount];

   for (i = 0; i < incount; i++) {
      array_of_requests[i] = req_array[i];
   }
   MPI_Waitsome(incount, array_of_requests, &outcount, array_of_indices, MPI_STATUSES_IGNORE);

   for (i = 0; i < incount; i++) {
      req_array[i] = array_of_requests[i];
   }
   delete [] array_of_requests;

   return outcount;
}

//______________________________________________________________________________
Int_t TRequest::TestSome(Int_t incount, TRequest req_array[], Int_t array_of_indices[], TStatus stat_array[])
{
   int i, outcount;
   MPI_Request *array_of_requests = new MPI_Request[incount];
   MPI_Status *array_of_statuses = new MPI_Status[incount];
   for (i = 0; i < incount; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   MPI_Testsome(incount, array_of_requests, &outcount, array_of_indices, array_of_statuses);
   for (i = 0; i < incount; i++) {
      req_array[i] = array_of_requests[i];
      stat_array[i] = array_of_statuses[i];
   }
   delete [] array_of_requests;
   delete [] array_of_statuses;
   return outcount;
}

//______________________________________________________________________________
Int_t TRequest::TestSome(Int_t incount, TRequest req_array[], Int_t array_of_indices[])
{
   Int_t i, outcount;
   MPI_Request *array_of_requests = new MPI_Request[incount];

   for (i = 0; i < incount; i++) {
      array_of_requests[i] = req_array[i];
   }
   MPI_Testsome(incount, array_of_requests, &outcount, array_of_indices, MPI_STATUSES_IGNORE);

   for (i = 0; i < incount; i++) {
      req_array[i] = array_of_requests[i];
   }
   delete [] array_of_requests;

   return outcount;
}

//______________________________________________________________________________
void TRequest::Cancel(void) const
{
   MPI_Cancel(const_cast<MPI_Request *>(&fRequest));
}

//______________________________________________________________________________
Bool_t TRequest::GetStatus(TStatus &status) const
{
   Int_t flag = 0;
   MPI_Status c_status;

   MPI_Request_get_status(fRequest, &flag, &c_status);
   if (flag) {
      status = c_status;
   }
   return Bool_t(flag);
}

//______________________________________________________________________________
Bool_t TRequest::GetStatus() const
{
   Int_t flag;

   MPI_Request_get_status(fRequest, &flag, MPI_STATUS_IGNORE);
   return Bool_t(flag);
}

//______________________________________________________________________________
void TPrequest::Start()
{
   MPI_Start(&fRequest);
}

//______________________________________________________________________________
void TPrequest::Startall(int count, TPrequest array_of_requests[])
{
   MPI_Request *mpi_requests = new MPI_Request[count];
   int i;
   for (i = 0; i < count; i++) {
      mpi_requests[i] = array_of_requests[i].fRequest;
   }
   MPI_Startall(count, mpi_requests);
   for (i = 0; i < count; i++) {
      array_of_requests[i].fRequest = mpi_requests[i] ;
   }
   delete [] mpi_requests;
}

//______________________________________________________________________________
TGrequest TGrequest::Start(Int_t(*query_fn)(void *, TStatus &), Int_t(*free_fn)(void *), Int_t(*cancel_fn)(void *, Bool_t), void *extra)
{
   MPI_Request grequest = 0;

   Intercept_data_t *new_extra = new Intercept_data_t;

   new_extra->id_cxx_query_fn = query_fn;
   new_extra->id_cxx_free_fn = free_fn;
   new_extra->id_cxx_cancel_fn = cancel_fn;
   new_extra->id_extra = extra;

   auto call_query_fn = [](void *extra_data, MPI_Status * status)->int {
      TGrequest::Intercept_data_t *data = (TGrequest::Intercept_data_t *)extra_data;
      TStatus stat;
      stat = *status;
      return data->id_cxx_query_fn(data->id_extra, stat);
   };

   auto call_free_fn = [](void *extra_data)->int {
      TGrequest::Intercept_data_t *data = (TGrequest::Intercept_data_t *)extra_data;
      return data->id_cxx_free_fn(data->id_extra);
   };

   auto call_cancel_fn = [](void *extra_data, int completed)->int {
      TGrequest::Intercept_data_t *data = (TGrequest::Intercept_data_t *)extra_data;
      return data->id_cxx_cancel_fn(data->id_extra, completed);
   };
   MPI_Grequest_start(call_query_fn,
                      call_free_fn,
                      call_cancel_fn,
                      new_extra, &grequest);
   return grequest;
}

//______________________________________________________________________________
void TGrequest::Complete()
{
   MPI_Grequest_complete(fRequest);
}

