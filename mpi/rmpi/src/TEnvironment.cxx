#include <Mpi/TEnvironment.h>
#include <Mpi/TIntraCommunicator.h>
#include <Mpi/TErrorHandler.h>
#include <iostream>
using namespace ROOT::Mpi;

TErrorHandler TEnvironment::fErrorHandler = TErrorHandler();

Int_t TEnvironment::fCompressionAlgorithm = 0;
Int_t TEnvironment::fCompressionLevel = 0;

TString TEnvironment::fStdOut = "";
TString TEnvironment::fStdErr = "";
Bool_t TEnvironment::fSyncOutput = kFALSE;
Int_t TEnvironment::fStdOutPipe[2] = {-1, -1};
Int_t TEnvironment::fStdErrPipe[2] = {-1, -1};
Int_t TEnvironment::fSavedStdErr = -1;
Int_t TEnvironment::fSavedStdOut = -1;

FILE *TEnvironment::fOutput = NULL;

// TODO: enable thread level and thread-safe for ROOT

//______________________________________________________________________________
/**
 * Default constructor to start the environment, initializes the MPI execution
environment
 * - ROOT::Mpi::THREAD_SINGLE: Only one thread will execute.
 * - ROOT::Mpi::THREAD_FUNNELED: The process may be multi-threaded, but only the
main thread
will make MPI calls (all MPI calls are ``funneled'' to the main thread).
 * - ROOT::Mpi::THREAD_SERIALIZED: The process may be multi-threaded, and
multiple threads
may make MPI calls, but only one at a time: MPI calls are not made concurrently
from two distinct threads (all MPI calls are ``serialized'').
 * - ROOT::Mpi::THREAD_MULTIPLE: Multiple threads may call MPI, with no
restrictions.
\param level is an integer with the thread type, default value THREAD_SINGLE is
equivalent to call the raw function MPI_Init
*/
TEnvironment::TEnvironment(Int_t level)
{
// export TMPDIR for OpenMPI at mac is required
// https://www.open-mpi.org/faq/?category=osx#startup-errors-with-open-mpi-2.0.x
#if defined(R__MACOSX) && defined(OPEN_MPI)
   gSystem->Setenv("TMPDIR", "/tmp");
#endif
   Int_t provided;
   MPI_Init_thread(NULL, NULL, level, &provided);

   if (IsInitialized()) {
      Int_t result;
      MPI_Comm_compare((MPI_Comm)COMM_WORLD, MPI_COMM_WORLD, &result);
      if (result == IDENT) COMM_WORLD.SetCommName("ROOT::Mpi::COMM_WORLD");
      ROOT_MPI_CHECK_CALL(MPI_Comm_set_errhandler, (MPI_COMM_WORLD, (MPI_Errhandler)fErrorHandler), &COMM_WORLD);
      InitSignalHandlers();
   } else {
      // TODO: added error handling here
   }
#if PYTHON_FOUND
   PyInit();
#endif
}

//______________________________________________________________________________
/**
 * Default constructor to start the environment, initializes the MPI execution
 * environment
 * - ROOT::Mpi::THREAD_SINGLE: Only one thread will execute.
 * - ROOT::Mpi::THREAD_FUNNELED: The process may be multi-threaded, but only the
 * main thread
 * will make MPI calls (all MPI calls are ``funneled'' to the main thread).
 * - ROOT::Mpi::THREAD_SERIALIZED: The process may be multi-threaded, and
 * multiple threads
 * may make MPI calls, but only one at a time: MPI calls are not made
 * concurrently from two distinct threads (all MPI calls are ``serialized'').
 * - ROOT::Mpi::THREAD_MULTIPLE: Multiple threads may call MPI, with no
 * restrictions.
 * \param argc integer with num of command line arguments
 * \param argv list of command line arguments
 * \param level is an integer with the thread type, default value THREAD_SINGLE
 * is equivalent to call the raw function MPI_Init
*/
TEnvironment::TEnvironment(Int_t argc, Char_t **argv, Int_t level)
{
// export TMPDIR for OpenMPI at mac is required
// https://www.open-mpi.org/faq/?category=osx#startup-errors-with-open-mpi-2.0.x
#if defined(R__MACOSX) && defined(OPEN_MPI)
   gSystem->Setenv("TMPDIR", "/tmp");
#endif
   Int_t provided;
   MPI_Init_thread(&argc, &argv, level, &provided);
   if (IsInitialized()) {
      Int_t result;
      ROOT_MPI_CHECK_CALL(MPI_Comm_compare, ((MPI_Comm)COMM_WORLD, MPI_COMM_WORLD, &result), &COMM_WORLD);
      ROOT_MPI_CHECK_CALL(MPI_Comm_set_errhandler, (MPI_COMM_WORLD, (MPI_Errhandler)fErrorHandler), &COMM_WORLD);
      InitSignalHandlers();
   } else {
      // TODO: added error handling here
   }
#if PYTHON_FOUND
   PyInit();
#endif
}

//______________________________________________________________________________
/**
 * Initialization of signal ahnlders to flush StdOut/StdErr
 */
void TEnvironment::InitSignalHandlers()
{
   fInterruptSignal = new TMpiSignalHandler(kSigInterrupt, *this);
   fTerminationSignal = new TMpiSignalHandler(kSigTermination, *this);
   fSigSegmentationViolationSignal = new TMpiSignalHandler(kSigSegmentationViolation, *this);
   gSystem->AddSignalHandler(fInterruptSignal);
   gSystem->AddSignalHandler(fTerminationSignal);
   gSystem->AddSignalHandler(fSigSegmentationViolationSignal);
}

//______________________________________________________________________________
TEnvironment::~TEnvironment()
{
   // if mpi's environment is initialized then finalize it
   if (!IsFinalized()) {
      Finalize();
   }
#if PYTHON_FOUND
   PyFinalize();
#endif
}

//______________________________________________________________________________
/**
 * Method to capture stdout/stderr into buffers, used to show synchronized
 * output for every process.
 */
void TEnvironment::InitCapture()
{
   if (fSyncOutput) {
      std::ios::sync_with_stdio();
      setvbuf(stdout, NULL, _IONBF, 0); // absolutely needed(flush not needed ;))
      setvbuf(stderr, NULL, _IONBF, 0); // absolutely needed

      /* save stdout/stderr for display later */
      fSavedStdOut = dup(STDOUT_FILENO);
      fSavedStdErr = dup(STDERR_FILENO);
      if (pipe(fStdOutPipe) != 0) { /* make a pipe for stdout*/
         return;
      }
      if (pipe(fStdErrPipe) != 0) { /* make a pipe for stdout*/
         return;
      }

      Long_t flags = fcntl(fStdOutPipe[0], F_GETFL);
      flags |= O_NONBLOCK;
      fcntl(fStdOutPipe[0], F_SETFL, flags);

      flags = fcntl(fStdErrPipe[0], F_GETFL);
      flags |= O_NONBLOCK;
      fcntl(fStdErrPipe[0], F_SETFL, flags);

      dup2(fStdOutPipe[1], STDOUT_FILENO); /* redirect stdout to the pipe */
      close(fStdOutPipe[1]);

      dup2(fStdErrPipe[1], STDERR_FILENO); /* redirect stderr to the pipe */
      close(fStdErrPipe[1]);
   }
   auto rank = -1;
   if (!IsFinalized()) {
      rank = COMM_WORLD.GetRank();
   }
   fStdOut += Form("-------  Rank %d OutPut  -------\n", rank);
}

//______________________________________________________________________________
/**
 * Method to stop capturing stdout/stderr into buffers.
 */
void TEnvironment::EndCapture()
{
   if (fSyncOutput) {
      Int_t buf_readed;
      Char_t ch;
      while (true) { /* read from pipe into buffer */
         fflush(stdout);
         std::cout.flush();
         buf_readed = read(fStdOutPipe[0], &ch, 1);
         if (buf_readed == 1)
            fStdOut += ch;
         else
            break;
      }

      while (true) { /* read from pipe into buffer */
         fflush(stderr);
         std::cerr.flush();
         buf_readed = read(fStdErrPipe[0], &ch, 1);
         if (buf_readed == 1)
            fStdErr += ch;
         else
            break;
      }

      dup2(fSavedStdOut, STDOUT_FILENO); /* reconnect stdout*/
      dup2(fSavedStdErr, STDERR_FILENO); /* reconnect stderr*/
   }
}

//______________________________________________________________________________
/**
 * Return stdout in string
 * \return string with the captured stdout.
 */
TString TEnvironment::GetStdOut()
{
   return fStdOut;
}

//______________________________________________________________________________
/**
 * Return stderr in string
 * \return string with the captured stderr.
 */
TString TEnvironment::GetStdErr()
{
   return fStdErr;
}

//______________________________________________________________________________
/**
 * Return true is synchronized output is enable
 * \return boolean true is synchronized output is enable.
 */
Bool_t TEnvironment::IsSyncOutput()
{
   return fSyncOutput;
}

//______________________________________________________________________________
void TEnvironment::Flush()
{
   if (fSyncOutput) {

      if (fOutput) {
         TString Output = fStdOut + fStdErr;
         if (fOutput == stdout) {
            write(fSavedStdOut, Output.Data(), Output.Length());
            fsync(fStdOutPipe[0]);
         }
         if (fOutput == stderr) {
            write(fSavedStdErr, Output.Data(), Output.Length());
            fsync(fStdErrPipe[0]);
         }
      } else {
         write(fSavedStdOut, fStdOut.Data(), fStdOut.Length());
         write(fSavedStdErr, fStdErr.Data(), fStdErr.Length());
         fsync(fStdOutPipe[0]);
         fsync(fStdErrPipe[0]);
      }
   } else {
      if (fOutput) {
         TString Output = fStdOut + fStdErr;
         fprintf(fOutput, "%s", Output.Data());
         fflush(fOutput);
      } else {
         fprintf(stdout, "%s", fStdOut.Data());
         fprintf(stderr, "%s", fStdErr.Data());
         fflush(stdout);
         fflush(stderr);
      }
   }
   ClearBuffers();
}

//______________________________________________________________________________
/**
 * Method to flush stdout/stderr given a communicator using ring algorithm
 * with an small delay to print synchronized.
 *\param comm Any non-null communicator object pointer
 */
void TEnvironment::Flush(TCommunicator *comm)
{
   Char_t dummy;
   if (comm->GetRank() != 0) {
      comm->Recv(dummy, comm->GetRank() - 1, comm->GetInternalTag());
      std::cout << fStdOut << std::flush;
      gSystem->Sleep(10 * comm->GetSize());
      std::cerr << fStdErr << std::flush;
      gSystem->Sleep(10 * comm->GetSize());
   } else {
   }
   comm->Send(dummy, (comm->GetRank() + 1) % comm->GetSize(), comm->GetInternalTag());

   if (comm->GetRank() == 0) {
      comm->Recv(dummy, comm->GetSize() - 1, comm->GetInternalTag());
      std::cout << fStdOut << std::flush;
      gSystem->Sleep(10 * comm->GetSize());
      std::cerr << fStdErr << std::flush;
      gSystem->Sleep(10 * comm->GetSize());
   }
   comm->Barrier();
   ClearBuffers();
}

//______________________________________________________________________________
/**
 *  Method to clear buffers on StdErr and StdOut
*/
void TEnvironment::ClearBuffers()
{
   fStdOut = "";
   fStdErr = "";
}

//______________________________________________________________________________
/**
 * Method to synchronize stdout/stderr output.
 * \param status enable/disable output synchronization
 * \param output FILE pointer to merge stdout and stderr
 * by default is merged in stdout but if output pointer is NULL stdout will be
 * printed asynchronous respect to stderr
*/
void TEnvironment::SyncOutput(Bool_t status, FILE *output)
{
   if (!fSyncOutput) {

      if (status) {
         fOutput = output;
         fSyncOutput = status;
         InitCapture();
      }
   } else {
      if (!status) {
         EndCapture();
         Flush();
         fOutput = output;
         fSyncOutput = status;
      }
   }
}

//______________________________________________________________________________
/**
 * Method to check if the communication system is finalized.
*/
Bool_t TEnvironment::IsFinalized()
{
   Int_t t;
   MPI_Finalized(&t);
   return Bool_t(t);
}

//______________________________________________________________
/**
 * Method to check if the communication system is initialized.
*/
Bool_t TEnvironment::IsInitialized()
{
   Int_t t;
   MPI_Initialized(&t);
   return (Bool_t)(t);
}

//______________________________________________________________________________
/**
 * Terminates MPI execution environment.
 */
void TEnvironment::Finalize()
{
   if (!IsFinalized()) {
      MPI_Finalize();
   }
   if (fSyncOutput) {
      EndCapture();
      Flush();
   }
}

//______________________________________________________________________________
/**
 * Gets the name of the processor
 */
TString TEnvironment::GetProcessorName()
{
   Char_t name[MAX_PROCESSOR_NAME];
   Int_t size;
   ROOT_MPI_CHECK_CALL(MPI_Get_processor_name, (name, &size), &COMM_WORLD);
   return TString(name, size);
}

//______________________________________________________________________________
/**
 * Returns the current level of thread support
 */
Int_t TEnvironment::GetThreadLevel()
{
   Int_t level;
   ROOT_MPI_CHECK_CALL(MPI_Query_thread, (&level), &COMM_WORLD);
   return level;
}

//______________________________________________________________________________
/**
 * True if calling thread is main thread (boolean).
 */
Bool_t TEnvironment::IsMainThread()
{
   Int_t status;
   ROOT_MPI_CHECK_CALL(MPI_Is_thread_main, (&status), &COMM_WORLD);
   return Bool_t(status);
}

//______________________________________________________________________________
/**
 *
 */
Int_t TEnvironment::GetCompressionAlgorithm()
{
   return fCompressionAlgorithm;
}

//______________________________________________________________________________
/**
 *
 */
Int_t TEnvironment::GetCompressionLevel()
{
   return fCompressionLevel;
}

//______________________________________________________________________________
/**
 *
 */
void TEnvironment::SetCompression(Int_t level, Int_t algorithm)
{
   fCompressionAlgorithm = algorithm;
   fCompressionLevel = level;
}
