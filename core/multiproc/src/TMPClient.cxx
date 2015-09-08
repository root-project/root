#include "TMPClient.h"
#include "TGuiFactory.h"
#include "TVirtualX.h"
#include "TSystem.h" //gSystem
#include "TROOT.h" //gROOT
#include "TObject.h"
#include "TServerSocket.h"
#include "EMPCode.h"
#include "TSocket.h"
#include "TMPServer.h"
#include "TCollection.h" //TIter
#include "TList.h"
#include "TError.h" //gErrorIgnoreLevel
#include <unistd.h> // close, fork
#include <sys/wait.h> // waitpid
#include <errno.h> //errno, used by socketpair
#include <sys/types.h> //socketpair
#include <sys/socket.h> //socketpair
#include <iostream>
#include <memory> //unique_ptr, shared_ptr
#include <list>

/// Class constructor.
TMPInterruptHandler::TMPInterruptHandler() : TSignalHandler(kSigInterrupt, kFALSE)
{
}

/// When SIGINT is received clean-up and quit the application
//TODO this should log somewhere that the server is being shut down
Bool_t TMPInterruptHandler::Notify()
{
   // logging does not work
   //gSystem->RedirectOutput(0);
   //std::cerr << "server shutting down on SIGINT" << std::endl;
   gSystem->Exit(0);
   return true;
}


//////////////////////////////////////////////////////////////////////////
/// Class constructor.
/// nWorkers is the number of children processes that will be created by
/// Fork, i.e. the number of workers that will be available during processing.
/// The default value is the total number of cores of the machine if available,
/// 2 otherwise.
TMPClient::TMPClient(unsigned nWorkers) : fIsParent(true), fServerPids(), fMon(), fNWorkers(0)
{
   // decide on number of workers
   if (nWorkers) {
      fNWorkers = nWorkers;
   } else {
      SysInfo_t si;
      if (gSystem->GetSysInfo(&si) == 0)
         fNWorkers = si.fCpus;
      else
         fNWorkers = 2;
   }
}


//////////////////////////////////////////////////////////////////////////
/// Class destructor.
/// This method is in charge of shutting down any remaining worker, 
/// closing off connections and reap the terminated children processes.
TMPClient::~TMPClient()
{
   Broadcast(EMPCode::kShutdownOrder);
   TList *l = fMon.GetListOfActives();
   l->Delete();
   delete l;
   l = fMon.GetListOfDeActives();
   l->Delete();
   delete l;
   fMon.RemoveAll();
   ReapServers();
}


//////////////////////////////////////////////////////////////////////////
/// This method forks the ROOT session into fNWorkers children processes.
/// The ROOT sessions spawned in this way will not have graphical
/// capabilities and will not read from standard input, but will be
/// connected to the original (interactive) session through TSockets.
/// The children processes' PIDs are added to the fServerPids vector.
/// The parent session can then communicate with the children using the
/// Broadcast and Send methods, and receive messages through Collect and
/// CollectOne.\n
/// After forking, the children processes will wait for events on server's
/// TSocket (e.g. messages sent by the parent session). When a message is
/// received, TMPServer::HandleInput is called if the code of the message
/// is above 1000, otherwise the unqualified (possibly overridden) version
/// of HandleInput is called, allowing classes that inherit from TMPServer
/// to manage their own protocol.
bool TMPClient::Fork(TMPServer *server)
{
   std::string basePath = "/tmp/ROOTMP-";

   //fork as many times as needed and save pids
   pid_t pid = 1; //must be positive to handle the case in which fNWorkers is 0
   int sockets[2]; //sockets file descriptors
   for (unsigned i = 0; i < fNWorkers; ++i) {
      //create socket pair
      int ret = socketpair(AF_UNIX, SOCK_STREAM, 0, sockets);
      if(ret != 0) {
         std::cerr << "[E][C] Could not create socketpair. Error n. " << errno << ". Now retrying.\n";
        --i;
        continue;
      }

      //fork
      pid = fork();

      if (!pid) {
         //child process, exit loop. sockets[1] is the fd that should be used
         break;
      } else {
         //parent process, create TSocket with current value of sockets[0]
         close(sockets[1]); //we don't need this
         TSocket *s = new TSocket(sockets[0], (std::to_string(pid)).c_str()); //TSocket's constructor with this signature seems much faster than TSocket(int fd)
         if(s && s->IsValid()) {
            fMon.Add(s);
            fServerPids.push_back(pid);
         } else {
            std::cerr << "[E][C] Could not connect to worker with pid " << pid << ". Giving up.\n";
            delete s;
         }
      }
   }

   if (pid) {
      //PARENT/CLIENT
      delete server; //the server is only needed by children processes
   } else {
      //CHILD/SERVER
      fIsParent = false;

      //override signal handler (make the servers exit on SIGINT)
      TSeqCollection *signalHandlers = gSystem->GetListOfSignalHandlers();
      TSignalHandler *sh = nullptr;
      if(signalHandlers && signalHandlers->GetSize() > 0)
         sh = (TSignalHandler*)signalHandlers->First();
      if(sh)
         gSystem->RemoveSignalHandler(sh);
      TMPInterruptHandler handler;
      handler.Add();

      //remove stdin from eventloop and close it
      TSeqCollection *fileHandlers = gSystem->GetListOfFileHandlers();
      if(fileHandlers) {
         TIter next(fileHandlers);
         TFileHandler *h = nullptr;
         while ((h = (TFileHandler *)next())) {
            if (h && h->GetFd() == 0) {
               gSystem->RemoveFileHandler(h);
               break;
            }
         }
      }
      close(0);

#ifndef R__WIN32
      //redirect output to /dev/null
      gSystem->RedirectOutput("/dev/null"); // we usually don't like servers to write on the main console
#endif

      //disable graphics
      //these instructions were copied from TApplication::MakeBatch
      gROOT->SetBatch();
      if (gGuiFactory != gBatchGuiFactory)
         delete gGuiFactory;
      gGuiFactory = gBatchGuiFactory;
#ifndef R__WIN32
      if (gVirtualX != gGXBatch)
         delete gVirtualX;
#endif
      gVirtualX = gGXBatch;

      //prepare server and add it to eventloop
      server->Init(sockets[1]);

      gSystem->Run();
   }

   return true;
}


//////////////////////////////////////////////////////////////////////////
///Send a message with the specified code to at most nMessages workers.
///If nMessages == 0, send specified code to all workers.
///The number of messages successfully sent is returned.
unsigned TMPClient::Broadcast(unsigned code, unsigned nMessages)
{
   if(!nMessages)
      nMessages = fNWorkers;
   unsigned count = 0;
   fMon.ActivateAll();

   //send message to all sockets
   std::unique_ptr<TList> l(fMon.GetListOfActives());
   TIter next(l.get());
   TSocket *s = nullptr;
   while ((s = (TSocket *)next()) && count < nMessages) {
      if(MPSend(s, code)) {
         fMon.DeActivate(s);
         ++count;
      } else {
         std::cerr << "[E] Could not send message to server\n";
      }
   }

   return count;
}


//////////////////////////////////////////////////////////////////////////
/// DeActivate a certain socket.
/// This does not remove it from the monitor: it will be reactivated by
/// the next call to Broadcast or possibly other methods.\n
/// A socket should be DeActivated by HandleInput when the corresponding
/// worker is done *for now* and we want to stop listening to this worker's
/// socket. If the worker is done _forever_, Remove should be used instead.
void TMPClient::DeActivate(TSocket *s)
{
   fMon.DeActivate(s);
}


//////////////////////////////////////////////////////////////////////////
/// Remove a certain socket from the monitor.
/// A socket should be Removed from the monitor by HandleInput when the
/// corresponding worker is done _forever_. For example, Remove is called
/// on sockets pertaining to workers which sent a kShutdownNotice code.
void TMPClient::Remove(TSocket *s)
{
   fMon.Remove(s);
   delete s;
}


//////////////////////////////////////////////////////////////////////////
/// Wait on worker processes.
/// This should actually not be a blocking operation, since ReapServers should
/// only be called when all server sessions have already finished their
/// jobs and quit. The waiting is done not to leave zombie processes
/// hanging around.
void TMPClient::ReapServers()
{
   for (auto &pid : fServerPids) {
      waitpid(pid, nullptr, 0);
   }
   fServerPids.clear();
}


//////////////////////////////////////////////////////////////////////////
/// TMPClient's implementation of HandleInput.
/// This method should be called upon receiving a message with a code >= 1000
/// (i.e. EMPCode). It handles the most generic types of messages.\n
/// Classes inheriting from TMPClient should implement their own HandleInput
/// function, that should be able to handle message codes specific to that
/// application.\n
void TMPClient::HandleMPCode(MPCodeBufPair& msg, TSocket *s)
{
   unsigned code = msg.first;
   //message contains server's pid. retrieve it
   char *str = new char[msg.second->BufferSize()];
   msg.second->ReadString(str, msg.second->BufferSize());

   if (code == EMPCode::kMessage) {
      std::cerr << "[I][C] message received: " << str << "\n";
   } else if (code == EMPCode::kError) {
      std::cerr << "[E][C] error message received:\n" << str << "\n";
   } else if (code == EMPCode::kShutdownNotice || code == EMPCode::kFatalError) {
      if(gDebug > 0) //generally users don't want to know this
         std::cerr << "[I][C] shutdown notice received from " << str << "\n";
      Remove(s);
   } else
      std::cerr << "[W][C] unknown code received. code=" << code << "\n";

   delete [] str;
}
