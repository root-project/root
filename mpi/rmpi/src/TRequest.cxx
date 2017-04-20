
#include<Mpi/TRequest.h>
#include<Mpi/TStatus.h>
#include<Mpi/TErrorHandler.h>

using namespace ROOT::Mpi;

//______________________________________________________________________________
TRequest::TRequest(): fRequest(MPI_REQUEST_NULL)
{
   fCallback = []() {};
}

//______________________________________________________________________________
TRequest::TRequest(const TRequest &obj): TObject(obj), fRequest(obj.fRequest)
{
   fCallback = []() {};
}


//______________________________________________________________________________
TRequest::TRequest(MPI_Request i) : fRequest(i)
{
   fCallback = []() {};
}


//______________________________________________________________________________
TRequest &TRequest::operator=(const TRequest &r)
{
   fRequest = r.fRequest;
   fCallback = r.fCallback;
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
   fCallback = [] {};
   return *this;
}

//______________________________________________________________________________
void TRequest::Wait(TStatus &status)
{
   ROOT_MPI_CHECK_CALL(MPI_Wait, (&fRequest, &status.fStatus), TRequest::Class_Name());
   //TODO:error control here if status is wrong
   if (fRequest == MPI_REQUEST_NULL) fCallback();
   else Warning(__FUNCTION__, "Resquest still lingers");
}

//______________________________________________________________________________
void TRequest::Wait()
{
   ROOT_MPI_CHECK_CALL(MPI_Wait, (&fRequest, MPI_STATUS_IGNORE), TRequest::Class_Name());
   //TODO:error control here if status is wrong
   if (fRequest == MPI_REQUEST_NULL) fCallback();
   else Warning(__FUNCTION__, "Resquest still lingers");
}

//______________________________________________________________________________
Bool_t TRequest::Test(TStatus &status)
{
   if (fRequest == MPI_REQUEST_NULL) Error(__FUNCTION__, "Calling Test on NULL resquest.");
   Int_t flag;
   ROOT_MPI_CHECK_CALL(MPI_Test, (&fRequest, &flag, &status.fStatus), TRequest::Class_Name());

   //TODO:error control here if status is wrong
   if (flag) fCallback();
   else Warning(__FUNCTION__, "Resquest test is not ready.");

   return (Bool_t)flag;
}

//______________________________________________________________________________
Bool_t TRequest::Test()
{
   if (fRequest == MPI_REQUEST_NULL) Error(__FUNCTION__, "Calling Test on NULL resquest.");
   Int_t flag;
   ROOT_MPI_CHECK_CALL(MPI_Test, (&fRequest, &flag, MPI_STATUS_IGNORE), TRequest::Class_Name());

   //TODO:error control here if status is wrong
   if (flag) fCallback();
   else Warning(__FUNCTION__, "Resquest test is not ready.");

   return (Bool_t)flag;
}

//______________________________________________________________________________
void TRequest::Free(void)
{
   ROOT_MPI_CHECK_CALL(MPI_Request_free, (&fRequest), TRequest::Class_Name());
}

//______________________________________________________________________________
Int_t TRequest::WaitAny(Int_t count, TRequest array[], TStatus &status)
{
   int index, i;
   MPI_Request *array_of_requests = new MPI_Request[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = array[i].fRequest;
   }
   ROOT_MPI_CHECK_CALL(MPI_Waitany, (count, array_of_requests, &index, &status.fStatus), TRequest::Class_Name());
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
   ROOT_MPI_CHECK_CALL(MPI_Waitany, (count, array_of_requests, &index, MPI_STATUS_IGNORE), TRequest::Class_Name());
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
   ROOT_MPI_CHECK_CALL(MPI_Testany, (count, array_of_requests, &index, &flag, &status.fStatus), TRequest::Class_Name());
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
   ROOT_MPI_CHECK_CALL(MPI_Testany, (count, array_of_requests, &index, &flag, MPI_STATUS_IGNORE), TRequest::Class_Name());
   for (i = 0; i < count; i++) {
      array[i] = array_of_requests[i];
   }
   delete [] array_of_requests;
   return Bool_t(flag);
}

//______________________________________________________________________________
void TRequest::WaitAll(Int_t count, TRequest req_array[], TStatus stat_array[])
{
   for (auto i = 0; i < count; i++) {
      req_array[i].Wait(stat_array[i]);
   }
}

//______________________________________________________________________________
void TRequest::WaitAll(Int_t count, TRequest req_array[])
{
   for (auto i = 0; i < count; i++) {
      req_array[i].Wait();
   }
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
   ROOT_MPI_CHECK_CALL(MPI_Testall, (count, array_of_requests, &flag, array_of_statuses), TRequest::Class_Name());
   for (i = 0; i < count; i++) {
      req_array[i] = array_of_requests[i];
      stat_array[i] = array_of_statuses[i];
      //TODO: added error hanling with status object here
      if (flag) req_array[i].fCallback();
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
   ROOT_MPI_CHECK_CALL(MPI_Testall, (count, array_of_requests, &flag, MPI_STATUSES_IGNORE), TRequest::Class_Name());

   for (i = 0; i < count; i++) {
      req_array[i] = array_of_requests[i];

      //TODO: added error hanling with status object here
      if (flag) req_array[i].fCallback();
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
   ROOT_MPI_CHECK_CALL(MPI_Waitsome, (incount, array_of_requests, &outcount,
                                      array_of_indices, array_of_statuses), TRequest::Class_Name());
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
   ROOT_MPI_CHECK_CALL(MPI_Waitsome, (incount, array_of_requests, &outcount, array_of_indices, MPI_STATUSES_IGNORE), TRequest::Class_Name());

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
   ROOT_MPI_CHECK_CALL(MPI_Testsome, (incount, array_of_requests, &outcount, array_of_indices, array_of_statuses), TRequest::Class_Name());
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
   ROOT_MPI_CHECK_CALL(MPI_Testsome, (incount, array_of_requests, &outcount, array_of_indices, MPI_STATUSES_IGNORE), TRequest::Class_Name());

   for (i = 0; i < incount; i++) {
      req_array[i] = array_of_requests[i];
   }
   delete [] array_of_requests;

   return outcount;
}

//______________________________________________________________________________
void TRequest::Cancel(void) const
{
   ROOT_MPI_CHECK_CALL(MPI_Cancel, (const_cast<MPI_Request *>(&fRequest)), TRequest::Class_Name());
}

//______________________________________________________________________________
Bool_t TRequest::GetStatus(TStatus &status) const
{
   Int_t flag = 0;
   MPI_Status c_status;

   ROOT_MPI_CHECK_CALL(MPI_Request_get_status, (fRequest, &flag, &c_status), TRequest::Class_Name());
   if (flag) {
      status = c_status;
   }
   return Bool_t(flag);
}

//______________________________________________________________________________
Bool_t TRequest::GetStatus() const
{
   Int_t flag;

   ROOT_MPI_CHECK_CALL(MPI_Request_get_status, (fRequest, &flag, MPI_STATUS_IGNORE), TRequest::Class_Name());
   return Bool_t(flag);
}

//______________________________________________________________________________
void TPrequest::Start()
{
   ROOT_MPI_CHECK_CALL(MPI_Start, (&fRequest), TPrequest::Class_Name());
}

//______________________________________________________________________________
void TPrequest::Startall(int count, TPrequest array_of_requests[])
{
   MPI_Request *mpi_requests = new MPI_Request[count];
   int i;
   for (i = 0; i < count; i++) {
      mpi_requests[i] = array_of_requests[i].fRequest;
   }
   ROOT_MPI_CHECK_CALL(MPI_Startall, (count, mpi_requests), TPrequest::Class_Name());
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
   ROOT_MPI_CHECK_CALL(MPI_Grequest_start, (call_query_fn,
                       call_free_fn,
                       call_cancel_fn,
                       new_extra, &grequest), TGrequest::Class_Name());
   return grequest;
}

//______________________________________________________________________________
void TGrequest::Complete()
{
   ROOT_MPI_CHECK_CALL(MPI_Grequest_complete, (fRequest), TGrequest::Class_Name());
}

