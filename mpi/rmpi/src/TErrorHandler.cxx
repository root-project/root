#include <Mpi/TErrorHandler.h>
#include <Mpi/TGroup.h>
#include <Mpi/TIntraCommunicator.h>

using namespace ROOT::Mpi;
Bool_t TErrorHandler::fVerbose = kFALSE;
//______________________________________________________________________________
TErrorHandler::TErrorHandler() : fErrorHandler(MPI_ERRORS_RETURN)
{
}

//______________________________________________________________________________
TErrorHandler::TErrorHandler(const TErrorHandler &err) : TObject(err), fErrorHandler(err.fErrorHandler)
{
}

//______________________________________________________________________________
/**
 * Converts an error code into an error class.
 * \param errcode Error code
 * \return Integer with error class
 */
Int_t TErrorHandler::GetErrorClass(Int_t errcode)
{
   Int_t eclass;
   MPI_Error_class(errcode, &eclass);
   return eclass;
}

//______________________________________________________________________________
/**
 * Returns a string for a given error code.
 * \param errcode Error code
 * \return String with error associated to error code.
 */
TString TErrorHandler::GetErrorString(Int_t errcode)
{
   Char_t *estring = new Char_t[MAX_ERROR_STRING];
   Int_t size;
   MPI_Error_string(errcode, estring, &size);
   return TString(estring, size);
}

//______________________________________________________________________________
/**
 * Associates a string with an error code or class.
 * \param errcode MPI error class, or an error code returned by an MPI routine
 * (integer).
 * \param msg Text that corresponds to the error code or class (string).
 */
void TErrorHandler::SetErrorString(Int_t errcode, const TString msg)
{
   MPI_Add_error_string(errcode, const_cast<Char_t *>(msg.Data()));
}

//______________________________________________________________________________
/**
 * Creates a new error class and returns its value
 * \return New error class (integer).
 */
Int_t TErrorHandler::CreateErrorClass()
{
   Int_t errcls;
   MPI_Add_error_class(&errcls);
   return errcls;
}

//______________________________________________________________________________
/**
 * Creates a new error code associated with errclass.
 * \param errclass MPI error class (integer).
 * \return Error code returned by an MPI routine or an MPI error class
 * (integer).
 */
Int_t TErrorHandler::CreateErrorCode(Int_t errclass)
{
   Int_t errcode;
   MPI_Add_error_code(errclass, &errcode);
   return errcode;
}

//______________________________________________________________________________
/**
 * Frees an MPI-style error handler.
 */
void TErrorHandler::Free()
{
   MPI_Errhandler_free(&fErrorHandler);
}

//______________________________________________________________________________
/**
* Specialized templated method to show a traceback and abort MPI execution for
* TGroup,
* The output has useful information for debug like rank, host, message etc..
* \param group TGroup object.
* \param function method/functions where the traceback was called.
* \param file file source
* \param line line number
* \param errcode MPI error code.
* \param _msg message with useful information.
*/

template <>
void TErrorHandler::TraceBack(const TGroup *group, const Char_t *function, const Char_t *file, Int_t line,
                              Int_t errcode, const Char_t *_msg)
{
   TString msg;
   if (TErrorHandler::IsVerbose()) {
      msg += Form("\nRank   = %d", group->GetRank());
      msg += Form("\nSize   = %d", group->GetSize());
      msg += Form("\nObject = %s", group->ClassName());
      msg += Form("\nHost   = %s", TEnvironment::GetProcessorName().Data());
   }

   msg += Form("\nCode = %d", errcode);
   msg += Form("\nName = %s", GetErrorString(errcode).Data());
   msg += Form("\nMessage = %s", _msg);
   msg += "\nAborting, finishing the remaining processes.";
   msg += "\n--------------------------------------------------------------------------\n";

   group->Error(Form("%s(...) %s[%d]", function, file, line), "%s", msg.Data());
   COMM_WORLD.Abort(errcode, kTRUE);
}

//______________________________________________________________________________
/**
* Specialized templated method to show a traceback and abort MPI execution for
* given class name,
* The output has useful information for debug like rank, host, message etc..
* \param class_name string with class name
* \param function method/functions where the traceback was called.
* \param file file source
* \param line line number
* \param errcode MPI error code.
* \param _msg message with useful information.
*/
template <>
void TErrorHandler::TraceBack(const Char_t *class_name, const Char_t *function, const Char_t *file, Int_t line,
                              Int_t errcode, const Char_t *_msg)
{
   TString msg;
   if (TErrorHandler::IsVerbose()) {
      msg += "\nUsing COMM_WORLD";
      msg += Form("\nRank   = %d", COMM_WORLD.GetRank());
      msg += Form("\nSize   = %d", COMM_WORLD.GetSize());
      msg += Form("\nObject = %s", class_name);
      msg += Form("\nHost   = %s", TEnvironment::GetProcessorName().Data());
   }

   msg += Form("\nCode = %d", errcode);
   msg += Form("\nName = %s", GetErrorString(errcode).Data());
   msg += Form("\nMessage = %s", _msg);
   msg += "\nAborting, finishing the remaining processes.";
   msg += "\n--------------------------------------------------------------------------\n";

   COMM_WORLD.Error(Form("%s(...) %s[%d]", function, file, line), "%s", msg.Data());
   COMM_WORLD.Abort(errcode, kTRUE);
}
