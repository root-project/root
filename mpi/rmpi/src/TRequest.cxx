
#include <Mpi/TRequest.h>
#include <Mpi/TStatus.h>
#include <Mpi/TErrorHandler.h>
#include <Mpi/TMpiMessage.h>
using namespace ROOT::Mpi;

//______________________________________________________________________________
TRequest::TRequest() : fRequest(MPI_REQUEST_NULL)
{
   fCallback = []() {};
}

//______________________________________________________________________________
TRequest::TRequest(const TRequest &obj) : TObject(obj), fRequest(obj.fRequest)
{
   fCallback = []() {};
}

//______________________________________________________________________________
TRequest::TRequest(MPI_Request i) : fRequest(i)
{
   fCallback = []() {};
}

//______________________________________________________________________________
TRequest::~TRequest()
{
}

//______________________________________________________________________________
TRequest &TRequest::operator=(const TRequest &r)
{
   fRequest = r.fRequest;
   fCallback = r.fCallback;
   return *this;
}

//______________________________________________________________________________
Bool_t TRequest::operator==(const TRequest &a)
{
   return (Bool_t)(fRequest == a.fRequest);
}

//______________________________________________________________________________
Bool_t TRequest::operator!=(const TRequest &a)
{
   return (Bool_t)(fRequest != a.fRequest);
}

//______________________________________________________________________________
TRequest &TRequest::operator=(const MPI_Request &i)
{
   fRequest = i;
   return *this;
}

//______________________________________________________________________________
/**
 * A call to ROOT::Mpi::TRequest::Wait  returns when the operation identified by
 * request is complete. If the communication object associated with this request
 * was created by a nonblocking send or receive call, then the object is
 * deallocated by the call to ROOT::Mpi::TRequest::Wait and the request handle
 * is set to ROOT::Mpi::REQUEST_NULL.
 *
 * The call returns, in status, information on the completed operation. The
 * content of the status object for a receive operation can be accessed as
 * described  in Section  3.2.5  of the MPI-1 Standard, "Return Status." The
 * status object for a send operation may be queried by a call to
 * ROOT::Mpi::TStatus::IsCancelled (see Section 3.8 of the MPI-1 Standard,
 * "Probe and Cancel").
 *
 * If your application does not need to examine the status field, you can save
 * resources by using the predefined constant ROOT::Mpi::STATUS_IGNORE as  a
 * special  value for the status argument.
 *
 * One is allowed to call ROOT::Mpi::TRequest::Wait with a null or inactive
 * request argument. In this case the operation returns immediately with empty
 * status.
 * \param status TStatus object passed by reference(status).
 */
void TRequest::Wait(TStatus &status)
{
   ROOT_MPI_CHECK_CALL(MPI_Wait, (&fRequest, &status.fStatus), TRequest::Class_Name());
   // TODO:error control here if status is wrong
   if (fRequest == MPI_REQUEST_NULL)
      fCallback();
   else
      Warning(__FUNCTION__, "Resquest still lingers");
}

//______________________________________________________________________________
/**
 * A call to ROOT::Mpi::TRequest::Wait  returns when the operation identified by
 * request is complete. If the communication object associated with this request
 * was created by a nonblocking send or receive call, then the object is
 * deallocated by the call to ROOT::Mpi::TRequest::Wait and the request handle
 * is set to ROOT::Mpi::REQUEST_NULL.
 *
 * The call returns, in status, information on the completed operation. The
 * content of the status object for a receive operation can be accessed as
 * described  in Section  3.2.5  of the MPI-1 Standard, "Return Status." The
 * status object for a send operation may be queried by a call to
 * ROOT::Mpi::TStatus::IsCancelled (see Section 3.8 of the MPI-1 Standard,
 * "Probe and Cancel").
 *
 * If your application does not need to examine the status field, you can save
 * resources by using the predefined constant ROOT::Mpi::STATUS_IGNORE as  a
 * special  value for the status argument.
 *
 * One is allowed to call ROOT::Mpi::TRequest::Wait with a null or inactive
 * request argument. In this case the operation returns immediately with empty
 * status.
 */
void TRequest::Wait()
{
   ROOT_MPI_CHECK_CALL(MPI_Wait, (&fRequest, MPI_STATUS_IGNORE), TRequest::Class_Name());
   // TODO:error control here if status is wrong
   if (fRequest == MPI_REQUEST_NULL)
      fCallback();
   else
      Warning(__FUNCTION__, "Resquest still lingers");
}

//______________________________________________________________________________
/**
 * Returns true if the operation identified by request is complete. In such a
 * case, the status object is set to contain information on the completed
 * operation; if the communication object was created by a nonblocking send or
 * receive, then it is deallocated and the request  handle  is  set  to
 * ROOT::Mpi::REQUEST_NULL. The call returns false, otherwise. In this case, the
 * value of the status object is undefined. ROOT::Mpi::TRequest::Test is a local
 * operation.
 *
 * The  return  status  object for a receive operation carries information that
 * can be accessed as described in Section 3.2.5 of the MPI-1 Standard, "Return
 * Status." The status object for a send operation carries information that can
 * be accessed by a call to ROOT::Mpi::TStatus::IsCancelled (see Section 3.8 of
 * the MPI-1  Standard,"Probe and Cancel").
 *
 * If  your  application  does not need to examine the status field, you can
 * save resources by using the predefined constant ROOT::Mpi::STATUS_IGNORE as a
 * special value for the status argument.
 *
 * One is allowed to call ROOT::Mpi::TRequest::Test  with a null or inactive
 * request argument. In such a case the operation returns  true and empty
 * status.
 * The functions ROOT::Mpi::TRequest::Wait  and ROOT::Mpi::TRequest::Test  can
 * be used to complete both sends and receives.
 * \param status Status object (status).
 * \return True if operation completed (logical).
 */
Bool_t TRequest::Test(TStatus &status)
{
   if (fRequest == MPI_REQUEST_NULL) Error(__FUNCTION__, "Calling Test on NULL resquest.");
   Int_t flag;
   ROOT_MPI_CHECK_CALL(MPI_Test, (&fRequest, &flag, &status.fStatus), TRequest::Class_Name());

   // TODO:error control here if status is wrong
   if (flag)
      fCallback();
   else
      Warning(__FUNCTION__, "Resquest test is not ready.");

   return (Bool_t)flag;
}

//______________________________________________________________________________
/**
 * Returns true if the operation identified by request is complete. In such a
 * case, the status object is set to contain information on the completed
 * operation; if the communication object was created by a nonblocking send or
 * receive, then it is deallocated and the request  handle  is  set  to
 * ROOT::Mpi::REQUEST_NULL. The call returns false, otherwise. In this case, the
 * value of the status object is undefined. ROOT::Mpi::TRequest::Test is a local
 * operation.
 *
 * The  return  status  object for a receive operation carries information that
 * can be accessed as described in Section 3.2.5 of the MPI-1 Standard, "Return
 * Status." The status object for a send operation carries information that can
 * be accessed by a call to ROOT::Mpi::TStatus::IsCancelled (see Section 3.8 of
 * the MPI-1  Standard,"Probe and Cancel").
 *
 * If  your  application  does not need to examine the status field, you can
 * save resources by using the predefined constant ROOT::Mpi::STATUS_IGNORE as a
 * special value for the status argument.
 *
 * One is allowed to call ROOT::Mpi::TRequest::Test  with a null or inactive
 * request argument. In such a case the operation returns  true and empty
 * status.
 * The functions ROOT::Mpi::TRequest::Wait  and ROOT::Mpi::TRequest::Test  can
 * be used to complete both sends and receives.
 * \return True if operation completed (logical).
 */
Bool_t TRequest::Test()
{
   if (fRequest == MPI_REQUEST_NULL) Error(__FUNCTION__, "Calling Test on NULL resquest.");
   Int_t flag;
   ROOT_MPI_CHECK_CALL(MPI_Test, (&fRequest, &flag, MPI_STATUS_IGNORE), TRequest::Class_Name());

   // TODO:error control here if status is wrong
   if (flag)
      fCallback();
   else
      Warning(__FUNCTION__, "Resquest test is not ready.");

   return (Bool_t)flag;
}

//______________________________________________________________________________
/**
 * This operation allows a request object to be deallocated without waiting for
 * the associated communication to complete.
 *
 * ROOT::Mpi::TRequest::Free marks the request object for deallocation and sets
 * request to ROOT::Mpi::REQUEST_NULL. Any ongoing communication that is
 * associated with the request will be allowed to complete. The request will be
 * deallocated only after its completion.
 *
 * <b>NOTE</b>
 * Once a request is freed by a call to ROOT::Mpi::TRequest::Free, it is not
 * possible to check for the successful completion of the associated
 * communication with calls to ROOT::Mpi::TRequest::Wait  or
 * ROOT::Mpi::TRequest::Test.  Also, if an error occurs subsequently during the
 * communication, an error code cannot be returned to the user -- such an error
 * must be treated as fatal. Questions arise as to how one knows when the
 * operations have completed when using ROOT::Mpi::TRequest::Free. Depending on
 * the  program  logic,  there may be other ways in which the program knows that
 * certain operations have completed and this makes usage of
 * ROOT::Mpi::TRequest::Free practical. For example, an active send request
 * could be freed when the logic of the program is such that the receiver sends
 * a reply to the message sent -- the arrival of the reply informs  the sender
 * that  the  send  has completed and the send buffer can be reused. An active
 * receive request should never be freed, as the receiver will have no way to
 * verify that the receive has completed and the receive buffer can be reused.
 */
void TRequest::Free(void)
{
   ROOT_MPI_CHECK_CALL(MPI_Request_free, (&fRequest), TRequest::Class_Name());
}

//______________________________________________________________________________
/**
 * This class can be used to wait for the completion of one out of several
 * requests.
 *
 * The array_of_requests list may contain null or inactive handles. If the list
 * contains no active handles (list has length zero or all entries are null or
 * inactive), then the call returns immediately with index =
 * ROOT::Mpi::UNDEFINED, and an empty status.
 *
 * The execution of ROOT::Mpi::TRequest::WaitAny(count, array_of_requests,
 * index, status) has the same effect as the execution of
 * ROOT::Mpi::TRequest::WaitAny(&array_of_requests[i], status), where i is the
 * value returned by index (unless the value of index is ROOT::Mpi::UNDEFINED).
 * ROOT::Mpi::TRequest::WaitAny with an array containing one active entry is
 * equivalent to ROOT::Mpi::TRequest::Wait
 *
 * If  your  application  does not need to examine the status field, you can
 * save resources by using the predefined constant ROOT::Mpi::STATUS_IGNORE as a
 * special value for the status argument.
 * \param count Array length (integer).
 * \param array Array of requests (array of handles).
 * \param status Status object (status).
 * \return Index of handle for operation that completed (integer).
 */
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
   delete[] array_of_requests;
   return index;
}

//______________________________________________________________________________
/**
 * This class can be used to wait for the completion of one out of several
 * requests.
 *
 * The array_of_requests list may contain null or inactive handles. If the list
 * contains no active handles (list has length zero or all entries are null or
 * inactive), then the call returns immediately with index =
 * ROOT::Mpi::UNDEFINED, and an empty status.
 *
 * If  your  application  does not need to examine the status field, you can
 * save resources by using the predefined constant ROOT::Mpi::STATUS_IGNORE as a
 * special value for the status argument.
 * \param count Array length (integer).
 * \param req_array Array of requests (array of handles).
 * \return Index of handle for operation that completed (integer).
 */
Int_t TRequest::WaitAny(Int_t count, TRequest req_array[])
{
   Int_t index, i;
   MPI_Request *array_of_requests = new MPI_Request[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   ROOT_MPI_CHECK_CALL(MPI_Waitany, (count, array_of_requests, &index, MPI_STATUS_IGNORE), TRequest::Class_Name());
   for (i = 0; i < count; i++) {
      req_array[i] = array_of_requests[i];
      if (array_of_requests[i] == MPI_REQUEST_NULL)
         req_array[i].fCallback();
      else
         req_array[i].Warning(__FUNCTION__, "Resquest still lingers");
   }
   delete[] array_of_requests;
   return index;
}

//______________________________________________________________________________
/**
 * Tests for completion of either one or none of the operations associated with
 * active handles. In the former case, it returns  true, returns in index the
 * index of this request in the array, and returns in status the status of that
 * operation; if the request was allocated by a nonblocking  communication  call
 * then the request is deallocated and the handle is set to
 * ROOT::Mpi::REQUEST_NULL. (The array is indexed from 0) In the latter case (no
 * operation completed), it returns  false, returns a value of
 * ROOT::Mpi::UNDEFINED in index, and status is undefined.
 *
 * The array may contain null or inactive handles. If the array contains no
 * active handles then the call returns immediately true, index =
 * ROOT::Mpi::UNDEFINED, and an empty status.
 *
 * If your application does not need to examine the status field, you can save
 * resources by using the predefined constant ROOT::Mpi::STATUS_IGNORE as  a
 * special  value for the status argument.
 *
 * \param count  Array length (integer).
 * \param req_array Array of requests (array of handles).
 * \param index Index of operation that completed, or ROOT::Mpi::UNDEFINED if
 * none completed (integer).
 * \param status Status object (status).
 * \return True if one of the operations is complete (logical).
 */
Bool_t TRequest::TestAny(Int_t count, TRequest req_array[], Int_t &index, TStatus &status)
{
   Int_t i, flag;
   MPI_Request *array_of_requests = new MPI_Request[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   ROOT_MPI_CHECK_CALL(MPI_Testany, (count, array_of_requests, &index, &flag, &status.fStatus), TRequest::Class_Name());
   for (i = 0; i < count; i++) {
      req_array[i] = array_of_requests[i];
   }
   if (flag)
      req_array[index].fCallback();
   else
      req_array[0].Warning(__FUNCTION__, "Resquest test is not ready.");

   delete[] array_of_requests;
   return (Bool_t)(flag != 0 ? true : false);
}

//______________________________________________________________________________
/**
 * Tests for completion of either one or none of the operations associated with
 * active handles. In the former case, it returns  true, returns in index the
 * index of this request in the array, and returns in status the status of that
 * operation; if the request was allocated by a nonblocking  communication  call
 * then the request is deallocated and the handle is set to
 * ROOT::Mpi::REQUEST_NULL. (The array is indexed from 0) In the latter case (no
 * operation completed), it returns  false, returns a value of
 * ROOT::Mpi::UNDEFINED in index, and status is undefined.
 *
 * The array may contain null or inactive handles. If the array contains no
 * active handles then the call returns immediately true, index =
 * ROOT::Mpi::UNDEFINED, and an empty status.
 *
 * If your application does not need to examine the status field, you can save
 * resources by using the predefined constant ROOT::Mpi::STATUS_IGNORE as  a
 * special  value for the status argument.
 *
 * \param count  Array length (integer).
 * \param req_array Array of requests (array of handles).
 * \param index Index of operation that completed, or ROOT::Mpi::UNDEFINED if
 * none completed (integer).
 * \return True if one of the operations is complete (logical).
 */
Bool_t TRequest::TestAny(Int_t count, TRequest req_array[], Int_t &index)
{
   Int_t i, flag;
   MPI_Request *array_of_requests = new MPI_Request[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   ROOT_MPI_CHECK_CALL(MPI_Testany, (count, array_of_requests, &index, &flag, MPI_STATUS_IGNORE),
                       TRequest::Class_Name());
   for (i = 0; i < count; i++) {
      req_array[i] = array_of_requests[i];
   }
   if (flag)
      req_array[index].fCallback();
   else
      req_array[0].Warning(__FUNCTION__, "Resquest test is not ready.");

   delete[] array_of_requests;
   return Bool_t(flag);
}

//______________________________________________________________________________
/**
 * Blocks  until  all communication operations associated with active handles in
 * the list complete, and returns the status of all these operations (this
 * includes the case where no handle in the list is active). Both arrays have
 * the same number of valid entries. The ith entry in array_of_statuses is set
 * to  the  return status  of  the  ith operation. Requests that were created by
 * nonblocking communication operations are deallocated, and the corresponding
 * handles in the array are set to ROOT::Mpi::REQUEST_NULL. The list may contain
 * null or inactive handles. The call sets to empty the status of each such
 * entry.
 *
 * When  one  or  more  of  the communications completed by a call to
 * ROOT::Mpi::TRequest::WaitAll fail, it is desirable to return specific
 * information on each communication. The method ROOT::Mpi::TRequest::WaitAll
 * will return in such case the error code ROOT::Mpi::ERR_IN_STATUS and will set
 * the error field of each status to a specific error code. This code will  be
 * ROOT::Mpi::SUCCESS if the specific communication completed; it will be
 * another specific error code if it failed; or it can be ROOT::Mpi::ERR_PENDING
 * if it has neither failed nor completed.
 * The method will return ROOT::Mpi::SUCCESS if no request had an error, or will
 * return another error code  if  it  failed  for other reasons (such as invalid
 * arguments). In such cases, it will not update the error fields of the
 * statuses.
 *
 * If  your  application does not need to examine the array_of_statuses field,
 * you can save resources by using the predefined constant
 * ROOT::Mpi::STATUSES_IGNORE can be used as a special value for the
 * array_of_statuses argument.
 *
 * \param count Array length (integer).
 * \param req_array Array of requests (array of handles).
 * \param stat_array Array of status objects (array of status).
 */
void TRequest::WaitAll(Int_t count, TRequest req_array[], TStatus stat_array[])
{
   Int_t i;
   MPI_Request *array_of_requests = new MPI_Request[count];
   MPI_Status *array_of_statuses = new MPI_Status[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   ROOT_MPI_CHECK_CALL(MPI_Waitall, (count, array_of_requests, array_of_statuses), TRequest::Class_Name());
   for (i = 0; i < count; i++) {
      req_array[i] = array_of_requests[i];
      stat_array[i] = array_of_statuses[i];
      if (array_of_requests[i] == MPI_REQUEST_NULL)
         req_array[i].fCallback();
      else
         req_array[i].Warning(__FUNCTION__, "Resquest still lingers");
   }
   delete[] array_of_requests;
   delete[] array_of_statuses;
}

//______________________________________________________________________________
/**
 * Blocks  until  all communication operations associated with active handles in
 * the list complete, and returns the status of all these operations (this
 * includes the case where no handle in the list is active). Both arrays have
 * the same number of valid entries. The ith entry in array_of_statuses is set
 * to  the  return status  of  the  ith operation. Requests that were created by
 * nonblocking communication operations are deallocated, and the corresponding
 * handles in the array are set to ROOT::Mpi::REQUEST_NULL. The list may contain
 * null or inactive handles. The call sets to empty the status of each such
 * entry.
 *
 * When  one  or  more  of  the communications completed by a call to
 * ROOT::Mpi::TRequest::WaitAll fail, it is desirable to return specific
 * information on each communication. The method ROOT::Mpi::TRequest::WaitAll
 * will return in such case the error code ROOT::Mpi::ERR_IN_STATUS and will set
 * the error field of each status to a specific error code. This code will  be
 * ROOT::Mpi::SUCCESS if the specific communication completed; it will be
 * another specific error code if it failed; or it can be ROOT::Mpi::ERR_PENDING
 * if it has neither failed nor completed.
 * The method will return ROOT::Mpi::SUCCESS if no request had an error, or will
 * return another error code  if  it  failed  for other reasons (such as invalid
 * arguments). In such cases, it will not update the error fields of the
 * statuses.
 *
 * If  your  application does not need to examine the array_of_statuses field,
 * you can save resources by using the predefined constant
 * ROOT::Mpi::STATUSES_IGNORE can be used as a special value for the
 * array_of_statuses argument.
 *
 * \param count Array length (integer).
 * \param req_array Array of requests (array of handles).
 */
void TRequest::WaitAll(Int_t count, TRequest req_array[])
{
   Int_t i;
   MPI_Request *array_of_requests = new MPI_Request[count];
   for (i = 0; i < count; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   ROOT_MPI_CHECK_CALL(MPI_Waitall, (count, array_of_requests, MPI_STATUSES_IGNORE), TRequest::Class_Name());
   for (i = 0; i < count; i++) {
      req_array[i] = array_of_requests[i];
      if (array_of_requests[i] == MPI_REQUEST_NULL)
         req_array[i].fCallback();
      else
         req_array[i].Warning(__FUNCTION__, "Resquest still lingers");
   }
   delete[] array_of_requests;
}

//______________________________________________________________________________
/**
 * Tests for the completion of all previously initiated communications in a
 * list.
 *
 * Returns true if all communications associated with active handles in the
 * array have completed (this includes the case where no handle in the list is
 * active). In this case, each status entry that corresponds to an active handle
 * request is set to the status of the corresponding communication; if the
 * request was  allocated  by a nonblocking communication call then it is
 * deallocated, and the handle is set to ROOT::Mpi::REQUEST_NULL. Each status
 * entry that corresponds to a null or inactive handle is set to empty.
 *
 * Otherwise, false is returned, no request is modified and the values of the
 * status entries are undefined. This is a local operation.
 *
 * If your application does not need to examine the array_of_statuses field, you
 * can save resources by using the predefined constant
 * ROOT::Mpi::STATUSES_IGNORE  can  be used as a special value for the
 * array_of_statuses argument.
 *
 * Errors that occurred during the execution of ROOT::Mpi::TRequest::TestAll are
 * handled in the same manner as errors in ROOT::Mpi::TRequest::WaitAll.
 *
 * <b>NOTE</b>
 * return is true only if all requests have completed. Otherwise, return is
 * false, and neither array_of_requests nor array_of_statuses is modified.
 * \param count Array length (integer).
 * \param req_array Array of requests (array of handles).
 * \param stat_array Array of status objects (array of status).
 * \return True if previously initiated communications are complete (logical.)
 */
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
      // TODO: added error hanling with status object here
      if (flag) req_array[i].fCallback();
   }
   delete[] array_of_requests;
   delete[] array_of_statuses;
   return Bool_t(flag);
}

//______________________________________________________________________________
/**
 * Tests for the completion of all previously initiated communications in a
 * list.
 *
 * Returns true if all communications associated with active handles in the
 * array have completed (this includes the case where no handle in the list is
 * active). In this case, each status entry that corresponds to an active handle
 * request is set to the status of the corresponding communication; if the
 * request was  allocated  by a nonblocking communication call then it is
 * deallocated, and the handle is set to ROOT::Mpi::REQUEST_NULL. Each status
 * entry that corresponds to a null or inactive handle is set to empty.
 *
 * Otherwise, false is returned, no request is modified and the values of the
 * status entries are undefined. This is a local operation.
 *
 * If your application does not need to examine the array_of_statuses field, you
 * can save resources by using the predefined constant
 * ROOT::Mpi::STATUSES_IGNORE  can  be used as a special value for the
 * array_of_statuses argument.
 *
 * Errors that occurred during the execution of ROOT::Mpi::TRequest::TestAll are
 * handled in the same manner as errors in ROOT::Mpi::TRequest::WaitAll.
 *
 * <b>NOTE</b>
 * return is true only if all requests have completed. Otherwise, return is
 * false, and neither array_of_requests nor array_of_statuses is modified.
 * \param count Array length (integer).
 * \param req_array Array of requests (array of handles).
 * \return True if previously initiated communications are complete (logical.)
 */
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

      // TODO: added error hanling with status object here
      if (flag) req_array[i].fCallback();
   }
   delete[] array_of_requests;

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
   ROOT_MPI_CHECK_CALL(MPI_Waitsome, (incount, array_of_requests, &outcount, array_of_indices, array_of_statuses),
                       TRequest::Class_Name());
   for (i = 0; i < incount; i++) {
      req_array[i] = array_of_requests[i];
      stat_array[i] = array_of_statuses[i];
      if (array_of_requests[i] == MPI_REQUEST_NULL)
         req_array[i].fCallback();
      else
         req_array[i].Warning(__FUNCTION__, "Resquest still lingers");
   }
   delete[] array_of_requests;
   delete[] array_of_statuses;
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
   ROOT_MPI_CHECK_CALL(MPI_Waitsome, (incount, array_of_requests, &outcount, array_of_indices, MPI_STATUSES_IGNORE),
                       TRequest::Class_Name());

   for (i = 0; i < incount; i++) {
      req_array[i] = array_of_requests[i];
      if (array_of_requests[i] == MPI_REQUEST_NULL)
         req_array[i].fCallback();
      else
         req_array[i].Warning(__FUNCTION__, "Resquest still lingers");
   }
   delete[] array_of_requests;

   return outcount;
}

//______________________________________________________________________________
Int_t TRequest::TestSome(Int_t incount, TRequest req_array[], Int_t array_of_indices[], TStatus stat_array[])
{
   Int_t i, outcount;
   MPI_Request *array_of_requests = new MPI_Request[incount];
   MPI_Status *array_of_statuses = new MPI_Status[incount];
   for (i = 0; i < incount; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   ROOT_MPI_CHECK_CALL(MPI_Testsome, (incount, array_of_requests, &outcount, array_of_indices, array_of_statuses),
                       TRequest::Class_Name());
   for (i = 0; i < incount; i++) {
      req_array[i] = array_of_requests[i];
      stat_array[i] = array_of_statuses[i];
   }

   for (i = 0; i < outcount; i++) {
      req_array[array_of_indices[i]].fCallback();
   }
   delete[] array_of_requests;
   delete[] array_of_statuses;
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
   ROOT_MPI_CHECK_CALL(MPI_Testsome, (incount, array_of_requests, &outcount, array_of_indices, MPI_STATUSES_IGNORE),
                       TRequest::Class_Name());

   for (i = 0; i < incount; i++) {
      req_array[i] = array_of_requests[i];
   }
   for (i = 0; i < outcount; i++) {
      req_array[array_of_indices[i]].fCallback();
   }
   delete[] array_of_requests;

   return outcount;
}

//______________________________________________________________________________
/**
 * This method allows pending communications to be canceled. This is required
 * for cleanup. Posting a send or a receive ties up user resources (send or
 * receive buffers), and a cancel may be needed to free these resources
 * gracefully.
 *
 * A call to this method marks for cancellation a pending, nonblocking
 * communication operation (send or receive). The cancel call is  local.  It
 * returns  immediately,  possibly before the communication is actually
 * canceled. It is still necessary to complete a communication that has been
 * marked for cancellation, using a call to ROOT::Mpi::TRequest::Free,
 * ROOT::Mpi::TRequest::Wait, or ROOT::Mpi::TRequest::Test (or any of the
 * derived operations).
 *
 * If a communication is marked for cancellation, then an
 * ROOT::Mpi::TRequest::Wait call for that communication is guaranteed to
 * return, irrespective of the activities  of  other processes  (i.e.,
 * ROOT::Mpi::TRequest::Wait  behaves  as  a local function); similarly if
 * ROOT::Mpi::TRequest::Test is repeatedly called in a busy wait loop for a
 * canceled communication, then ROOT::Mpi::TRequest::Test will eventually be
 * successful.
 *
 * ROOT::Mpi::TRequest::Cancel can be used to cancel a communication that uses a
 * persistent request (see Section 3.9 in the MPI-1 Standard, "Persistent
 * Communication  Requests") in  the same way it is used for nonpersistent
 * requests. A successful cancellation cancels the active communication, but not
 * the request itself. After the call to ROOT::Mpi::TRequest::Cancel and the
 * subsequent call to ROOT::Mpi::TRequest::Wait or ROOT::Mpi::TRequest::Test,
 * the request becomes inactive and can be activated for a new communication.
 *
 * The successful cancellation of a buffered send frees the buffer space
 * occupied by the pending message.
 * Either the cancellation succeeds or the communication succeeds, but not both.
 * If a send is marked for cancellation, then it must be the case that  either
 * the send  completes  normally,  in which case the message sent is received at
 * the destination process, or that the send is successfully canceled, in which
 * case no part of the message is received at the destination. Then, any
 * matching receive has to be satisfied by another send. If a receive is marked
 * for  cancellation, then it must be the case that either the receive completes
 * normally, or that the receive is successfully canceled, in which case no part
 * of the receive buffer is altered. Then, any matching send has to be satisfied
 * by another receive.
 *
 * If the operation has been canceled, then information to that effect will be
 * returned in the status argument of the operation that completes the
 * communication.
 *
 * <b>NOTE</b>
 * The primary expected use of ROOT::Mpi::TRequest::Cancel is in multi-buffering
 * schemes, where speculative IRecvs are made.  When the computation completes,
 * some  of  these requests may remain; using ROOT::Mpi::TRequest::Cancel allows
 * the user to cancel these unsatisfied requests.
 */
void TRequest::Cancel(void) const
{
   ROOT_MPI_CHECK_CALL(MPI_Cancel, (const_cast<MPI_Request *>(&fRequest)), TRequest::Class_Name());
}

//______________________________________________________________________________
/**
 * Access information associated with a request without freeing the request.
 * Return true if the operation is complete or false if it is not complete. If
 * the operation is complete, it returns in status the request status. It does
 * not deallocate or inactivate the request; a subsequent call to test, wait, or
 * free should be executed with that request.
 *
 * If your application does not need to examine the status field, you can save
 * resources by using the predefined constant ROOT::Mpi::STATUS_IGNORE as  a
 * special  value for the status argument.
 * \param status TStatus object if return is true (status).
 * \return true if the operation is completed.
 */
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
/**
 * Access information associated with a request without freeing the request.
 * Return true if the operation is complete or false if it is not complete. If
 * the operation is complete, it returns in status the request status. It does
 * not deallocate or inactivate the request; a subsequent call to test, wait, or
 * free should be executed with that request.
 *
 * If your application does not need to examine the status field, you can save
 * resources by using the predefined constant ROOT::Mpi::STATUS_IGNORE as  a
 * special  value for the status argument.
 * \return true if the operation is completed.
 */
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
void TPrequest::Startall(int count, TPrequest req_array[])
{
   MPI_Request *array_of_requests = new MPI_Request[count];
   int i;
   for (i = 0; i < count; i++) {
      array_of_requests[i] = req_array[i].fRequest;
   }
   ROOT_MPI_CHECK_CALL(MPI_Startall, (count, array_of_requests), TPrequest::Class_Name());
   for (i = 0; i < count; i++) {
      req_array[i].fRequest = array_of_requests[i];
   }
   delete[] array_of_requests;
}

//______________________________________________________________________________
/**
 * Starts a generalized request and returns a handle to it in request.
 * The  syntax  and  meaning  of the callback functions are listed below. All
 * callback functions are passed the extra_state argument that was associated
 * with the request by the starting call MPI_Grequest_start. This can be used to
 * maintain user-defined state for the request.
 *
 * <b>The query function is</b>
 *
 * Int_t(*Query_fn)(void *extra_state, TStatus &extra_state)
 *
 * The query_fn function computes the status that should be returned for the
 * generalized request. The status also includes  information  about
 * successful/unsuccessful cancellation of the request (result to be returned by
 * ROOT::Mpi::TStatus::IsCancelled).
 *
 * The  query_fn function is invoked by the
 * ROOT::Mpi::TRequest::{Wait|Test}{Any|Some|All} call that completed the
 * generalized request associated with this callback.
 * The callback function is also invoked by calls to
 * ROOT::Mpi::TRequest::GetStatus if the request is complete when the call
 * occurs. In both cases, the callback is passed a reference to  the
 * corresponding  status  variable passed by the user to the MPI call. If the
 * user provided ROOT::Mpi::STATUS_IGNORE or ROOT::Mpi::STATUSES_IGNORE to the
 * MPI function that causes query_fn to be called, then MPI will pass a valid
 * status object to query_fn, and this status will be ignored upon return of the
 * callback function.
 *
 * Note  that  query_fn  is  invoked only after ROOT::Mpi::Grequest:Complete is
 * called on the request; it may be invoked several times for the same
 * generalized request.
 * Note also that a call to ROOT::Mpi::TRequest::{Wait|Test}{some|all} may cause
 * multiple invocations of query_fn callback functions, one for each
 * generalized  request  that  is completed by the MPI call. The order of these
 * invocations is not specified by MPI.
 *
 * <b>The free function is</b>
 *
 * Int_t(*Free_fn)(void * extra_state)
 *
 * The free_fn callback function is invoked to clean up user-allocated resources
 * when the generalized request is freed.
 *
 * The  free_fn  function  is  invoked by the
 * ROOT::Mpi::TRequest::{Wait|Test}{Any|Some|All} call that completed the
 * generalized request associated with this callback. free_fn is invoked after
 * the call to query_fn for the same request. However, if the MPI call completed
 * multiple generalized requests, the order in which free_fn callback functions
 * are invoked is not specified by MPI.
 *
 * The  free_fn  callback  is  also  invoked for generalized requests that are
 * freed by a call to ROOT::Mpi::TRequest::Free (no call to
 * ROOT::Mpi::TRequest::{Wait|Test}{Any|Some|All} will occur for such a
 * request). In this case, the callback function will be called either in the
 * MPI call ROOT::Mpi::TRequest::Free(request) or in  the  MPI  call
 * ROOT::Mpi::Grequest:Complete(request),  whichever happens last. In other
 * words, in this case the actual freeing code is executed as soon as both calls
 * ROOT::Mpi::Grequest:Free and ROOT::Mpi::Grequest:Complete) have occurred. The
 * request is not deallocated until after free_fn completes. Note that free_fn
 * will be invoked only once per request by a correct program.
 *
 * <b>The cancel function is</b>
 *
 * Int_t(*Cancel_fn)(void *extra_state, Bool_t complete)
 *
 * The  cancel_fn function is invoked to start the cancellation of a generalized
 * request. It is called by ROOT::Mpi::Request::Cancel(request). MPI passes to
 * the callback function complete=true if ROOT::Mpi::Grequest:Complete has
 * already been called on the request, and complete=false otherwise.
 *
 * \param query_fn Callback function invoked when request status is queried
 * (function).
 * \param free_fn Callback function invoked when request is freed (function).
 * \param cancel_fn Callback function invoked when request is canceled
 * (function).
 * \param extra Extra state.
 * \return Generalized request (handle).
 */
TGrequest TGrequest::Start(Int_t (*query_fn)(void *, TStatus &), Int_t (*free_fn)(void *),
                           Int_t (*cancel_fn)(void *, Bool_t), void *extra)
{
   MPI_Request grequest = 0;

   Intercept_data_t *new_extra = new Intercept_data_t;

   new_extra->id_cxx_query_fn = query_fn;
   new_extra->id_cxx_free_fn = free_fn;
   new_extra->id_cxx_cancel_fn = cancel_fn;
   new_extra->id_extra = extra;

   auto call_query_fn = [](void *extra_data, MPI_Status *status) -> int {
      TGrequest::Intercept_data_t *data = (TGrequest::Intercept_data_t *)extra_data;
      TStatus stat;
      stat = *status;
      return data->id_cxx_query_fn(data->id_extra, stat);
   };

   auto call_free_fn = [](void *extra_data) -> int {
      TGrequest::Intercept_data_t *data = (TGrequest::Intercept_data_t *)extra_data;
      return data->id_cxx_free_fn(data->id_extra);
   };

   auto call_cancel_fn = [](void *extra_data, int completed) -> int {
      TGrequest::Intercept_data_t *data = (TGrequest::Intercept_data_t *)extra_data;
      return data->id_cxx_cancel_fn(data->id_extra, completed);
   };
   ROOT_MPI_CHECK_CALL(MPI_Grequest_start, (call_query_fn, call_free_fn, call_cancel_fn, new_extra, &grequest),
                       TGrequest::Class_Name());
   return grequest;
}

//______________________________________________________________________________
/**
 * Reports that a generalized request is complete.
 *
 * Informs  MPI  that the operations represented by the generalized request
 * request are complete. A call to ROOT::Mpi::Request::Wait will return, and a
 * call to ROOT::Mpi::Request::Test will return true only after a call to
 * ROOT::Mpi::Request::TGrequest::Complete has declared that these operations
 * are complete.
 *
 * MPI imposes no restrictions on the code executed by the callback functions.
 * However, new nonblocking operations should be defined so that the general
 * semantic rules about MPI calls such as ROOT::Mpi::Request::Test,
 * ROOT::Mpi::Request::Free, or ROOT::Mpi::Request::Cancel still hold. For
 * example, all these calls are supposed to  be  local  and  nonblocking.
 *
 * Therefore,  the callback functions query_fn, free_fn, or cancel_fn should
 * invoke blocking MPI communication calls only if the context is such that
 * these calls are guaranteed to return in finite time. Once
 * ROOT::Mpi::Request::Cancel has been invoked, the canceled operation should
 * complete in finite time, regardless of the  state  of other  processes  (the
 * operation has acquired "local" semantics). It should either succeed or fail
 * without side-effects. The user should guarantee these same properties for
 * newly defined operations.
 */
void TGrequest::Complete()
{
   ROOT_MPI_CHECK_CALL(MPI_Grequest_complete, (fRequest), TGrequest::Class_Name());
   if (fRequest == MPI_REQUEST_NULL)
      fCallback();
   else
      Warning(__FUNCTION__, "Resquest still lingers");
}
