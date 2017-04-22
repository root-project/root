#include<Mpi/TEnvironment.h>
#include<Mpi/TIntraCommunicator.h>
#include<Mpi/TErrorHandler.h>
using namespace ROOT::Mpi;

TErrorHandler TEnvironment::fErrorHandler = TErrorHandler();

//TODO: enable thread level and thread-safe for ROOT

//______________________________________________________________________________
/**
Default constructor to start the environment, initializes the MPI execution environment
THREAD_SINGLE: Only one thread will execute.
THREAD_FUNNELED: The process may be multi-threaded, but only the main thread will make MPI calls (all MPI calls are ``funneled'' to the main thread).
THREAD_SERIALIZED: The process may be multi-threaded, and multiple threads may make MPI calls, but only one at a time: MPI calls are not made concurrently from two distinct threads (all MPI calls are ``serialized'').
THREAD_MULTIPLE: Multiple threads may call MPI, with no restrictions.
\param level is an integer with the thread type, default value THREAD_SINGLE is equivalent to call the raw function MPI_Init
*/
TEnvironment::TEnvironment(Int_t level): fSyncOutput(kFALSE)
{
   Int_t provided;
   MPI_Init_thread(NULL, NULL, level, &provided);

   if (IsInitialized()) {
      Int_t result;
      MPI_Comm_compare((MPI_Comm)COMM_WORLD, MPI_COMM_WORLD, &result);
      if (result == IDENT) COMM_WORLD.SetCommName("ROOT::Mpi::COMM_WORLD");
      ROOT_MPI_CHECK_CALL(MPI_Comm_set_errhandler, (MPI_COMM_WORLD, (MPI_Errhandler)fErrorHandler), &COMM_WORLD);
   } else {
      //TODO: added error handling here
   }
}

//______________________________________________________________________________
/**
Default constructor to start the environment, initializes the MPI execution environment
THREAD_SINGLE: Only one thread will execute.
THREAD_FUNNELED: The process may be multi-threaded, but only the main thread will make MPI calls (all MPI calls are ``funneled'' to the main thread).
THREAD_SERIALIZED: The process may be multi-threaded, and multiple threads may make MPI calls, but only one at a time: MPI calls are not made concurrently from two distinct threads (all MPI calls are ``serialized'').
THREAD_MULTIPLE: Multiple threads may call MPI, with no restrictions.
\param argc integer with num of command line arguments
\param argv list of command line arguments
\param level is an integer with the thread type, default value THREAD_SINGLE is equivalent to call the raw function MPI_Init
*/
TEnvironment::TEnvironment(Int_t argc, Char_t **argv, Int_t level): fSyncOutput(kFALSE)
{
   Int_t provided;
   MPI_Init_thread(&argc, &argv, level, &provided);
   if (IsInitialized()) {
      Int_t result;
      ROOT_MPI_CHECK_CALL(MPI_Comm_compare, ((MPI_Comm)COMM_WORLD, MPI_COMM_WORLD, &result), &COMM_WORLD);
      ROOT_MPI_CHECK_CALL(MPI_Comm_set_errhandler, (MPI_COMM_WORLD, (MPI_Errhandler)fErrorHandler), &COMM_WORLD);
   } else {
      //TODO: added error handling here
   }
}

//______________________________________________________________________________
TEnvironment::~TEnvironment()
{
   //if mpi's environment is initialized then finalize it
   if (!IsFinalized()) {
      Finalize();
   }
}

//______________________________________________________________________________
void TEnvironment::InitCapture()
{
   if (fSyncOutput) {
      std::ios::sync_with_stdio();
      setvbuf(stdout, NULL, _IONBF, 0);  // absolutely needed(flush not needed ;))
      setvbuf(stderr, NULL, _IONBF, 0);  // absolutely needed

      /* save stdout/stderr for display later */
      fSavedStdOut = dup(STDOUT_FILENO);
      fSavedStdErr = dup(STDERR_FILENO);
      if (pipe(fStdOutPipe) != 0) {           /* make a pipe for stdout*/
         return;
      }
      if (pipe(fStdErrPipe) != 0) {           /* make a pipe for stdout*/
         return;
      }

      Long_t flags = fcntl(fStdOutPipe[0], F_GETFL);
      flags |= O_NONBLOCK;
      fcntl(fStdOutPipe[0], F_SETFL, flags);

      flags = fcntl(fStdErrPipe[0], F_GETFL);
      flags |= O_NONBLOCK;
      fcntl(fStdErrPipe[0], F_SETFL, flags);

      dup2(fStdOutPipe[1], STDOUT_FILENO);   /* redirect stdout to the pipe */
      close(fStdOutPipe[1]);

      dup2(fStdErrPipe[1], STDERR_FILENO);   /* redirect stderr to the pipe */
      close(fStdErrPipe[1]);
   }
}

//______________________________________________________________________________
void TEnvironment::EndCapture()
{
   if (fSyncOutput) {
      Int_t buf_readed;
      Char_t ch;
      while (true) { /* read from pipe into buffer */
         fflush(stdout);
         buf_readed = read(fStdOutPipe[0], &ch, 1);
         if (buf_readed == 1) fStdOut += ch;
         else break;
      }

      while (true) { /* read from pipe into buffer */
         buf_readed = read(fStdErrPipe[0], &ch, 1);
         if (buf_readed == 1) fStdErr += ch;
         else break;
      }

      dup2(fSavedStdOut, STDOUT_FILENO);  /* reconnect stdout*/
      dup2(fSavedStdErr, STDERR_FILENO);  /* reconnect stderr*/
   }
}

//______________________________________________________________________________
void TEnvironment::Flush()
{
   if (fSyncOutput) {
      write(fSavedStdOut, fStdOut.Data(), fStdOut.Length());
      write(fSavedStdErr, fStdErr.Data(), fStdErr.Length());
      fsync(fStdOutPipe[0]);
      fsync(fStdErrPipe[0]);
   } else {
      fprintf(stdout, "%s", fStdOut.Data());
      fprintf(stderr, "%s", fStdErr.Data());
      fflush(stdout);
      fflush(stderr);
   }
   ClearBuffers();
}

//______________________________________________________________________________
void TEnvironment::ClearBuffers()
{
   fStdOut = "";
   fStdErr = "";
}

//______________________________________________________________________________
void TEnvironment::SyncOutput(Bool_t status)
{
   fSyncOutput = status;
   InitCapture();
}

//______________________________________________________________________________
Bool_t TEnvironment::IsFinalized()
{
   Int_t t;
   MPI_Finalized(&t);
   return Bool_t(t);
}

//______________________________________________________________
Bool_t TEnvironment::IsInitialized()
{
   Int_t t;
   MPI_Initialized(&t);
   return (Bool_t)(t);
}


//______________________________________________________________________________
void TEnvironment::Finalize()
{
   auto rank = -1;
   if (!IsFinalized()) {
      rank = COMM_WORLD.GetRank();
      //Finalize the mpi's environment
      MPI_Finalize();
   }
   if (fSyncOutput) {
      EndCapture();
      printf("-------  Rank %d OutPut  -------\n", rank);
      Flush();
   }
}

//______________________________________________________________________________
TString TEnvironment::GetProcessorName()
{
   Char_t name[MAX_PROCESSOR_NAME];
   Int_t size;
   ROOT_MPI_CHECK_CALL(MPI_Get_processor_name, (name, &size), &COMM_WORLD);
   return TString(name, size);
}

//______________________________________________________________________________
Int_t TEnvironment::GetThreadLevel()
{
   Int_t level;
   ROOT_MPI_CHECK_CALL(MPI_Query_thread, (&level), &COMM_WORLD);
   return level;
}

//______________________________________________________________________________
Bool_t TEnvironment::IsMainThread()
{
   Int_t status;
   ROOT_MPI_CHECK_CALL(MPI_Is_thread_main, (&status), &COMM_WORLD);
   return Bool_t(status);
}
