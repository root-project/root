/*
   C++ implementation of Bonjour services.  Code is based
   on the implementation of TBonjour* classes written by
   Fons Rademakers for the ROOT Framework.
*/

#include <arpa/inet.h>
#include <net/if.h>
#include <sys/select.h>
#include <cstdlib>
#include "Xrd/XrdConfig.hh"
#include "XrdOuc/XrdOucBonjour.hh"
#include "XrdOuc/XrdOucFactoryBonjour.hh"
#include "XrdSys/XrdSysError.hh"
#include "Xrd/XrdInet.hh"
#include "Xrd/XrdProtLoad.hh"

/******************************************************************************/
/*                      G l o b a l   V a r i a b l e s                       */
/******************************************************************************/

extern XrdConfig    XrdConf;              // Defined in XrdMain.cc
extern XrdSysError  XrdLog;               // Defined in XrdMain.cc
extern XrdInet     *XrdNetTCP[];          // Defined by config

/******************************************************************************/
/*        B o n j o u r   s e r v i c e s : r e g i s t r a t i o n           */
/******************************************************************************/

void XrdOucAppleBonjour::RegisterReply(DNSServiceRef ref, DNSServiceFlags flags,
                                       DNSServiceErrorType error, const char * name,
                                       const char * regtype, const char * domain,
                                       void * context)
{
   XrdOucBonjourRegisteredEntry * entry;
   entry = (XrdOucBonjourRegisteredEntry *)context;

   // When this callback is called, it comes with the final results of the
   // procedure and the registered names if there was a renaming due to name
   // collisions with other services.
   if (error != kDNSServiceErr_NoError) {
      ((XrdOucAppleBonjour &)getInstance()).ListOfRegistrations.remove(entry);
      delete entry->record;
      free(entry);
      XrdLog.Emsg("OucBonjour", error, "complete the registration callback");
      return;
   }

   // Update the registered information.
   entry->record->SetServiceName(name);
   entry->record->SetRegisteredType(regtype);
   entry->record->SetReplyDomain(domain);
}

int XrdOucAppleBonjour::RegisterService(XrdOucBonjourRecord &record, unsigned short port)
{
   XrdOucBonjourRegisteredEntry * entry;
   DNSServiceErrorType err;
   fd_set descriptors;
   int sockfd;
   struct timeval timeout;

   // Get the default port.
   if (port == 0)
      port = (XrdNetTCP[0] == XrdNetTCP[XrdProtLoad::ProtoMax]
              ?  -(XrdNetTCP[0]->Port()) : XrdNetTCP[0]->Port());

   // Store information on local list.
   entry = (XrdOucBonjourRegisteredEntry *)malloc(sizeof(XrdOucBonjourRegisteredEntry));
   if (!entry)
      return -1;

   entry->record = new XrdOucBonjourRecord(record);
   entry->port = port;

   // Start the registration procedure.
   err = DNSServiceRegister(&(entry->bonjourRef),
                            0, // No flags
                            0,
                            record.GetServiceName(),
                            record.GetRegisteredType(),
                            record.GetReplyDomain(),
                            NULL, // This host
                            htons(port),
                            record.GetTXTRecordLength(),
                            record.GetTXTRecordData(),
                            RegisterReply,
                            entry);

   if (err != kDNSServiceErr_NoError) {
      XrdLog.Emsg("OucBonjour", err, "rRegigster service", record.GetRegisteredType());
      XrdLog.Emsg("OucBonjour", err, "launch the registration");
      // Free memory.
      delete entry->record;
      free(entry);
      return -1;
   }

   // Wait for the callback. CAUTION: this call is blocking and may stuck you
   // thread. This will call the callback function.
   sockfd = DNSServiceRefSockFD(entry->bonjourRef);
   // Set the fd_set to start select()
   FD_ZERO(&descriptors);
   FD_SET(sockfd, &descriptors);
   timeout.tv_sec = TIMEOUT;
   timeout.tv_usec = 0;
   // Wait for the response.
   if (select(sockfd + 1, &descriptors, NULL, NULL, &timeout) > 0) {
      err = DNSServiceProcessResult(entry->bonjourRef);
      if (err != kDNSServiceErr_NoError) {
         XrdLog.Emsg("OucBonjour", err, "process the registration procedure");
         return -1;
      }
   } else {
      XrdLog.Emsg("OucBonjour", err, "wait to the registration response");
      return -1;
   }

   ListOfRegistrations.push_back(entry);
   return 0;
}

/******************************************************************************/
/*           B o n j o u r   s e r v i c e s : d i s c o v e r y              */
/******************************************************************************/

void * XrdOucAppleBonjour::BrowseEventLoopThread(void * context)
{
   int sockfd;
   int selRes;
   int stopEventLoop = 0;
   DNSServiceRef serviceRef;
   DNSServiceErrorType err;
   fd_set descriptors;
   XrdOucBonjourSubscribedEntry * callbackID;

   callbackID = (XrdOucBonjourSubscribedEntry *)context;

   // This thread is responsible for receiving the updates from the mDNS daemon.
   // We are using a select() based event-loop because is more extensible if we
   // would like to register for more services than one.
   XrdSysThread::SetCancelOn();
   XrdSysThread::SetCancelAsynchronous();

   // Start the DNS service browsing on the network.
   err = DNSServiceBrowse(&serviceRef,
                          0,
                          0,
                          callbackID->serviceType->c_str(),
                          NULL,
                          BrowseReply,
                          context);

   // If there are no errors, launch the event loop.
   if (err != kDNSServiceErr_NoError) {
      XrdLog.Emsg("OucBonjour", err, "launch the discovery process");
      return NULL; // Thread ends.
   }

   // Launch the event loop.
   sockfd = DNSServiceRefSockFD(serviceRef);

   while (!stopEventLoop) {
      // Set the fd_set to start select()
      FD_ZERO(&descriptors);
      FD_SET(sockfd, &descriptors);

      // Wait until there is something to do
      selRes = select(sockfd + 1, &descriptors, NULL, NULL, NULL);

      // Process the new result.
      if (selRes > 0) {
         if (FD_ISSET(sockfd, &descriptors)) {
            // This function will call the appropiate callback to process the
            // event, in this case the BrowseReply static method.
            err = DNSServiceProcessResult(serviceRef);
            if (err != kDNSServiceErr_NoError) {
               XrdLog.Emsg("OucBonjour", err, "process an event in event loop");
               stopEventLoop = 1;
            }
         }
      } else if (selRes == -1) {
         XrdLog.Emsg("OucBonjour", "The select() call failed");
         stopEventLoop = 1;
      }
   }

   XrdLog.Emsg("OucBonjour", "The browsing event loop has ended unexpectedly");
   return NULL; // Thread ends.
}

void XrdOucAppleBonjour::BrowseReply(DNSServiceRef ref, DNSServiceFlags flags,
                                     uint32_t interfaceIndex, DNSServiceErrorType error,
                                     const char * name, const char * regtype,
                                     const char * domain, void * context)
{
   // Comes with the context parameter to avoid abusing of getInstance()
   XrdOucAppleBonjour *instance;
   XrdOucBonjourNode *node;
   XrdOucBonjourSubscribedEntry * callbackID;
   XrdOucBonjourResolutionEntry * nodeAndCallback;

   callbackID = (XrdOucBonjourSubscribedEntry *)context;

   if (error != kDNSServiceErr_NoError) {
      XrdLog.Emsg("OucBonjour", error, "complete the browse callback");
      return;
   }

   // Get the context (the XrdOucBonjour object which holds the lists of nodes).
   instance = &XrdOucAppleBonjour::getInstance();

   // Process the node. First let us know what type of update is.
   if (flags & kDNSServiceFlagsAdd) {
      // ADD a new node to the list.
      node = new XrdOucBonjourNode(name, regtype, domain);
      nodeAndCallback = (XrdOucBonjourResolutionEntry *)malloc(sizeof(XrdOucBonjourResolutionEntry));
      nodeAndCallback->node = node;
      nodeAndCallback->callbackID = callbackID;

      // Start resolution of the name.
      instance->ResolveNodeInformation(nodeAndCallback);

      // We are going to wait to add the node until it is completely resolved.
      //instance->LockNodeList();
      //instance->ListOfNodes.push_back(node);
      //instance->UnLockNodeList();

      XrdLog.Say("------ XrdOucBonjour: discovered a new node: ", name);
   } else {
      // REMOVE this node from the list.
      XrdOucAppleBonjourSearchNode predicate(name);

      instance->LockNodeList();
      instance->ListOfNodes.remove_if(predicate);
      instance->UnLockNodeList();

      XrdLog.Say("------ XrdOucBonjour: the node ", name, " went out the network");

      // Notify updates if there wont be more updates in a short period of time.
      if (!(flags & kDNSServiceFlagsMoreComing))
         callbackID->callback(callbackID->context);
   }
}

/******************************************************************************/
/*       B o n j o u r   s e r v i c e s : n o t i f i c a t i o n s          */
/******************************************************************************/

int XrdOucAppleBonjour::SubscribeForUpdates(const char * servicetype,
                                            XrdOucBonjourUpdateCallback callback,
                                            void * context)
{
   pthread_t thread;
   XrdOucBonjourSubscribedEntry * callbackID = (XrdOucBonjourSubscribedEntry *)malloc(sizeof(XrdOucBonjourSubscribedEntry));
   callbackID->callback = callback;
   callbackID->context = context;
   callbackID->serviceType = new XrdOucString(servicetype);

   // Lauch the new browsing thread.
   return XrdSysThread::Run(&thread, BrowseEventLoopThread, callbackID);
}

/******************************************************************************/
/*          B o n j o u r   s e r v i c e s : r e s o l u t i o n             */
/******************************************************************************/

void XrdOucAppleBonjour::ResolveReply(DNSServiceRef ref, DNSServiceFlags flags,
                                      uint32_t interfaceIndex, DNSServiceErrorType error,
                                      const char * fullname, const char * hostname,
                                      uint16_t port, uint16_t txtLen,
                                      const unsigned char * txtVal, void * context)
{
   XrdOucAppleBonjour * instance;
   XrdOucBonjourResolutionEntry * nodeAndCallback;

   if (error != kDNSServiceErr_NoError) {
      XrdLog.Emsg("OucBonjour", error, "complete the resolve callback");
      return;
   }

   nodeAndCallback = static_cast<XrdOucBonjourResolutionEntry *>(context);

   // Copy the information of resolution results to the node.
   XrdLog.Say("------ XrdOucBonjour: resolved FQDN of new node: ", hostname);
   nodeAndCallback->node->SetHostName(hostname);
   nodeAndCallback->node->SetPort(ntohs(port));

   // Also, copy the TXT values.
   nodeAndCallback->node->GetBonjourRecord().AddRawTXTRecord((const char *)txtVal);

   // Get the context (the XrdOucBonjour object which holds the lists of nodes).
   instance = &XrdOucAppleBonjour::getInstance();

   // We are ready to add the node to the list and invoke the callback.
   instance->LockNodeList();
   instance->ListOfNodes.push_back(nodeAndCallback->node);
   instance->UnLockNodeList();

   // Notify updates if there wont be more updates in a short period of time.
   if (!(flags & kDNSServiceFlagsMoreComing))
      nodeAndCallback->callbackID->callback(nodeAndCallback->callbackID->context);

   free(nodeAndCallback);
}

int XrdOucAppleBonjour::ResolveNodeInformation(XrdOucBonjourResolutionEntry * nodeAndCallback)
{
   DNSServiceErrorType err;
   DNSServiceRef serviceRef;
   fd_set descriptors;
   int sockfd;
   struct timeval timeout;

   // Launch the resolution procedure.
   err = DNSServiceResolve(&serviceRef,
                           0,
                           0,
                           nodeAndCallback->node->GetBonjourRecord().GetServiceName(),
                           nodeAndCallback->node->GetBonjourRecord().GetRegisteredType(),
                           nodeAndCallback->node->GetBonjourRecord().GetReplyDomain(),
                           ResolveReply,
                           nodeAndCallback);

   // Check for errors
   if (err != kDNSServiceErr_NoError) {
      XrdLog.Emsg("OucBonjour", err, "start the resolution procedure");
      return -1;
   }

   // Wait for the callback. CAUTION: this call is blocking and may stuck you
   // thread. This will call the callback function.
   sockfd = DNSServiceRefSockFD(serviceRef);
   // Set the fd_set to start select()
   FD_ZERO(&descriptors);
   FD_SET(sockfd, &descriptors);
   timeout.tv_sec = TIMEOUT;
   timeout.tv_usec = 0;
   // Wait for the response.
   if (select(sockfd + 1, &descriptors, NULL, NULL, &timeout) > 0) {
      err = DNSServiceProcessResult(serviceRef);
      // Cancel the resolution process since we must have the data yet.
      DNSServiceRefDeallocate(serviceRef);
      if (err != kDNSServiceErr_NoError) {
         XrdLog.Emsg("OucBonjour", err, "process the resolution procedure");
         return -1;
      }
   } else {
      XrdLog.Emsg("OucBonjour", err, "wait to the resolution response");
      return -1;
   }

   // We finished this OK.
   return 0;
}

/******************************************************************************/
/*        C o n s t r u c t o r s   &   S i n g l e t o n   s t u f f         */
/******************************************************************************/

bool XrdOucAppleBonjour::XrdOucAppleBonjourSearchNode::operator()(XrdOucBonjourNode * value)
{
   return strcmp(value->GetBonjourRecord().GetServiceName(), ServiceName) == 0;
}

XrdOucAppleBonjour * XrdOucAppleBonjour::_Instance = NULL;

XrdSysMutex XrdOucAppleBonjour::SingletonMutex;

XrdOucAppleBonjour::XrdOucAppleBonjour()
{
   char *env = new char[22];
   strcpy(env, "AVAHI_COMPAT_NOWARN=1");
   putenv(env);
}

XrdOucAppleBonjour::~XrdOucAppleBonjour() { }

// In this case, to get a portable solution, we are not using any platform
// specific keyword (like volatile on Win), so, to minimize the cost of this
// function (mainly, gaining the lock) is highly recommended that any client of
// this class stores a local reference to the singleton instance in order to
// minimize the number of queries to the lock.
XrdOucAppleBonjour &XrdOucAppleBonjour::getInstance()
{
   // At the moment this object is destroyed, the singleton instance will be
   // deleted.
   static XrdOucAppleBonjourSingletonCleanup cleanGuard;

   SingletonMutex.Lock();
   if (!_Instance)
      _Instance = new XrdOucAppleBonjour();
   SingletonMutex.UnLock();

   return *_Instance;
}

XrdOucAppleBonjour::XrdOucAppleBonjourSingletonCleanup::~XrdOucAppleBonjourSingletonCleanup()
{
   SingletonMutex.Lock();
   if (_Instance) {
      delete XrdOucAppleBonjour::_Instance;
      XrdOucAppleBonjour::_Instance = NULL;
   }
   SingletonMutex.UnLock();
}
