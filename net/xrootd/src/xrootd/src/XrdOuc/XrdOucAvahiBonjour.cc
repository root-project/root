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

void XrdOucAvahiBonjour::EntryGroupReply(AvahiEntryGroup *g,
                                         AvahiEntryGroupState state,
                                         void *userdata)
{
   XrdOucBonjourRegisteredEntry * entry = (XrdOucBonjourRegisteredEntry *)userdata;

   switch (state) {
      case AVAHI_ENTRY_GROUP_COLLISION: {
         char *n;
         // There was a collision between other service and ours, let's pick
         // another name.
         n = avahi_alternative_service_name(entry->record->GetServiceName());
         entry->record->SetServiceName(n);
         XrdLog.Emsg("OucBonjour", "Renaming service to ", entry->record->GetServiceName());
         avahi_free(n);
         // Recreate services.
         RegisterEntries(entry);
         break;
      }

      case AVAHI_ENTRY_GROUP_FAILURE:
         // Some failure has occured.
         XrdLog.Emsg("OucBonjour", avahi_strerror(avahi_client_errno(avahi_entry_group_get_client(g))));
         break;
      case AVAHI_ENTRY_GROUP_ESTABLISHED:
      case AVAHI_ENTRY_GROUP_UNCOMMITED:
      case AVAHI_ENTRY_GROUP_REGISTERING:
         // Do not do anything in this case.
         break;
      default:
         XrdLog.Emsg("OucBonjour", "Invalid Avahi group creation response callback");
   }
}

void XrdOucAvahiBonjour::RegisterEntries(XrdOucBonjourRegisteredEntry * entry)
{
   int ret;
   char *n;
   AvahiStringList *list = NULL;

   // Create the entry group if it is necessary
   if (!entry->avahiRef.avahiEntryGroup) {
      entry->avahiRef.avahiEntryGroup = avahi_entry_group_new(entry->avahiRef.avahiClient,
                                                              EntryGroupReply,
                                                              (void *)entry);
      if (!entry->avahiRef.avahiEntryGroup) {
         XrdLog.Emsg("OucBonjour", "Unable to register service entries");
         XrdLog.Emsg("OucBonjour", avahi_strerror(avahi_client_errno(entry->avahiRef.avahiClient)));
         return;
      }
   }

   // If the entry group is empty, fill it with the service details.
   if (avahi_entry_group_is_empty(entry->avahiRef.avahiEntryGroup)) {
      // Get TXT in Avahi format.
      list = entry->record->GetTXTAvahiList();
      // Add the service described by the XrdBonjour register.
      if ((ret = avahi_entry_group_add_service_strlst(entry->avahiRef.avahiEntryGroup, // Avahi entry group
                                                      AVAHI_IF_UNSPEC, // All the interfaces
                                                      AVAHI_PROTO_INET, // Both IPv4 and IPv6
                                                      (AvahiPublishFlags)0, // No flags
                                                      entry->record->GetServiceName(),
                                                      entry->record->GetRegisteredType(),
                                                      entry->record->GetReplyDomain(),
                                                      NULL, // Let the system put the hostname
                                                      entry->port,
                                                      list)) < 0) {
         if (ret == AVAHI_ERR_COLLISION) {
            // There was a collision between other service and ours, let's pick
            // another name.
            n = avahi_alternative_service_name(entry->record->GetServiceName());
            entry->record->SetServiceName(n);
            XrdLog.Emsg("OucBonjour", "Renaming service to ", entry->record->GetServiceName());
            avahi_free(n);
            // Recreate services.
            RegisterEntries(entry);
            return;
         }
      }

      // After adding the resource, tell the server that we want to commit the
      // registration on the network.
      if ((ret = avahi_entry_group_commit(entry->avahiRef.avahiEntryGroup)) < 0) {
         XrdLog.Emsg("OucBonjour", "Unable to commit entry registration ", avahi_strerror(ret));
      }

      // Final clean.
      if (list)
         avahi_string_list_free(list);
   }
}

void XrdOucAvahiBonjour::RegisterReply(AvahiClient *c,
                                       AvahiClientState state,
                                       void * userdata)
{
   XrdOucBonjourRegisteredEntry * entry = (XrdOucBonjourRegisteredEntry *)userdata;

   // Assign the client structure.
   entry->avahiRef.avahiClient = c;

   switch (state) {
      case AVAHI_CLIENT_S_RUNNING:
         // The registration was OK, let's put some services on the server.
         RegisterEntries(entry);
         break;
      case AVAHI_CLIENT_FAILURE:
         XrdLog.Emsg("OucBonjour", avahi_strerror(avahi_client_errno(c)));
         break;
      case AVAHI_CLIENT_S_COLLISION:
      case AVAHI_CLIENT_S_REGISTERING:
         // We must reset the entry group since the name has changed.
         if (entry->avahiRef.avahiEntryGroup)
            avahi_entry_group_reset(entry->avahiRef.avahiEntryGroup);
         break;
      case AVAHI_CLIENT_CONNECTING:
         // Do nothing in this case.
         break;
      default:
         XrdLog.Emsg("OucBonjour", "Invalid Avahi register response callback");
   }
}

int XrdOucAvahiBonjour::RegisterService(XrdOucBonjourRecord &record, unsigned short port)
{
   XrdOucBonjourRegisteredEntry * entry;
   int err;

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
   entry->avahiRef.avahiEntryGroup = NULL;

   // Start the registration procedure.
   entry->avahiRef.avahiClient = avahi_client_new(avahi_simple_poll_get(poller),
                                                  (AvahiClientFlags)0,
                                                  RegisterReply,
                                                  (void *)entry,
                                                  &err);

   if (!entry->avahiRef.avahiClient) {
      XrdLog.Emsg("OucBonjour", err, "Regigster service", record.GetRegisteredType());
      XrdLog.Emsg("OucBonjour", err, avahi_strerror(err));
      // Free memory.
      delete entry->record;
      free(entry);
      return -1;
   }

   // With Avahi it is not necessary to wait actively for the callback.

   ListOfRegistrations.push_back(entry);
   return 0;
}

/******************************************************************************/
/*           B o n j o u r   s e r v i c e s : d i s c o v e r y              */
/******************************************************************************/

void XrdOucAvahiBonjour::BrowseReply(AvahiServiceBrowser *b,
                                     AvahiIfIndex interface,
                                     AvahiProtocol protocol,
                                     AvahiBrowserEvent event,
                                     const char *name,
                                     const char *type,
                                     const char *domain,
                                     AvahiLookupResultFlags flags,
                                     void* userdata)
{
   XrdOucBonjourSubscribedEntry * callbackID;
   XrdOucAvahiBonjour *instance;
   XrdOucAvahiBonjourSearchNode predicate(name);
   XrdOucBonjourResolutionEntry * toResolve;

   callbackID = (XrdOucBonjourSubscribedEntry *)userdata;

   // Get the context (the XrdOucBonjour object which holds the lists of nodes).
   instance = &XrdOucAvahiBonjour::getInstance();

   switch (event) {
      case AVAHI_BROWSER_FAILURE:
         XrdLog.Emsg("OucBonjour", avahi_strerror(avahi_client_errno(callbackID->client)));
         return;

      case AVAHI_BROWSER_NEW:

        instance->LockNodeList();
        // ADD a new node to the list.
        toResolve = (XrdOucBonjourResolutionEntry *)malloc(sizeof(XrdOucBonjourResolutionEntry));
        toResolve->node = new XrdOucBonjourNode(name, type, domain);
        toResolve->callbackID = callbackID;

         XrdLog.Say("------ XrdOucBonjour: discovered a new node: ", name);

         // Start resolution of the name.
         if (!(avahi_service_resolver_new(callbackID->client, interface, protocol, name, type, domain, AVAHI_PROTO_INET, (AvahiLookupFlags)0, ResolveReply, toResolve)))
            XrdLog.Emsg("OucBonjour", avahi_strerror(avahi_client_errno(callbackID->client)));

        // Wait until the node is rsolved to insert it on the list AND invoke the callback.
         instance->UnLockNodeList();

         break;

      case AVAHI_BROWSER_REMOVE:

         // REMOVE this node from the list.
         instance->LockNodeList();
         instance->ListOfNodes.remove_if(predicate);
         instance->UnLockNodeList();

         XrdLog.Say("------ XrdOucBonjour: the node ", name, " went out the network");

         // Invoke the callback inmediately.
         callbackID->callback(callbackID->context);

         break;

      case AVAHI_BROWSER_ALL_FOR_NOW:
      case AVAHI_BROWSER_CACHE_EXHAUSTED:
         break;
   }

}

void XrdOucAvahiBonjour::ClientReply(AvahiClient *c,
                                     AvahiClientState state,
                                     void * userdata)
{
   if (state == AVAHI_CLIENT_FAILURE) {
      XrdLog.Emsg("OucBonjour", avahi_strerror(avahi_client_errno(c)));
   }
}

void * XrdOucAvahiBonjour::BrowseEventLoopThread(void * context)
{
   AvahiSimplePoll *simple_poll = NULL;
   AvahiServiceBrowser *sb = NULL;
   int error;
   XrdOucBonjourSubscribedEntry * callbackID;

   callbackID = (XrdOucBonjourSubscribedEntry *)context;

   // Allocate main loop object
   if (!(simple_poll = avahi_simple_poll_new())) {
      XrdLog.Emsg("OucBonjour", "Failed to create the poller discovery object");
      return NULL;
   }

   // Allocate a new client
   callbackID->client = avahi_client_new(avahi_simple_poll_get(simple_poll), (AvahiClientFlags)0, ClientReply, callbackID, &error);

   // Check wether creating the client object succeeded
   if (!callbackID->client) {
      XrdLog.Emsg("OucBonjour", error, avahi_strerror(error));
      avahi_simple_poll_free(simple_poll);
      return NULL;
   }

   // Create the service browser
   if (!(sb = avahi_service_browser_new(callbackID->client, AVAHI_IF_UNSPEC, AVAHI_PROTO_INET, callbackID->serviceType->c_str(), NULL, (AvahiLookupFlags)0, BrowseReply, callbackID))) {
      XrdLog.Emsg("OucBonjour", avahi_strerror(avahi_client_errno(callbackID->client)));
      avahi_client_free(callbackID->client);
      avahi_simple_poll_free(simple_poll);
      return NULL;
   }

   // Run the main loop
   avahi_simple_poll_loop(simple_poll);

   XrdLog.Emsg("OucBonjour", "Event loop thread terminated abnormally");

   return NULL; // Thread ends.
}

/******************************************************************************/
/*       B o n j o u r   s e r v i c e s : n o t i f i c a t i o n s          */
/******************************************************************************/

int XrdOucAvahiBonjour::SubscribeForUpdates(const char * servicetype,
                                            XrdOucBonjourUpdateCallback callback,
                                            void * context)
{
   pthread_t thread;
   XrdOucBonjourSubscribedEntry * callbackID = (XrdOucBonjourSubscribedEntry *)malloc(sizeof(XrdOucBonjourSubscribedEntry));
   callbackID->callback = callback;
   callbackID->context = context;
   callbackID->serviceType = new XrdOucString(servicetype);
   callbackID->client = NULL;

   // Lauch the new browsing thread.
   return XrdSysThread::Run(&thread, BrowseEventLoopThread, callbackID);
}

/******************************************************************************/
/*          B o n j o u r   s e r v i c e s : r e s o l u t i o n             */
/******************************************************************************/

void XrdOucAvahiBonjour::ResolveReply(AvahiServiceResolver *r,
                                      AvahiIfIndex interface,
                                      AvahiProtocol protocol,
                                      AvahiResolverEvent event,
                                      const char *name,
                                      const char *type,
                                      const char *domain,
                                      const char *host_name,
                                      const AvahiAddress *address,
                                      uint16_t port,
                                      AvahiStringList *txt,
                                      AvahiLookupResultFlags flags,
                                      void* userdata)
{
   XrdOucBonjourSubscribedEntry * callbackID;
   XrdOucBonjourResolutionEntry * toResolve;
   XrdOucAvahiBonjour *instance;
   AvahiStringList * iterator;
   char * key, * value, address_str[AVAHI_ADDRESS_STR_MAX];
   size_t size;

   toResolve = static_cast<XrdOucBonjourResolutionEntry *>(userdata);
   callbackID = toResolve->callbackID;

   switch (event) {
      case AVAHI_RESOLVER_FAILURE:
         XrdLog.Emsg("OucBonjour", avahi_strerror(avahi_client_errno(avahi_service_resolver_get_client(r))));
         break;

      case AVAHI_RESOLVER_FOUND:
         instance = &XrdOucAvahiBonjour::getInstance();
         instance->LockNodeList();
	 // Check if the resolver gethostbyname() function supports mDNS lookups.
	 if (avahi_nss_support()) {
            // Copy the information of resolution results to the node since the
	    // name can be resolved.
            // NOTE: The signature of this method is compliant with Avahi developer docs, but, actually, the
            //       proper hostname is caming in the name parameter, instead of on the hostname one, that holds
            //       the service name.
            XrdLog.Say("------ XrdOucBonjour: resolved FQDN of new node: ", name);
            toResolve->node->SetHostName(name);
         } else {
	    // Save the address directly to improve name resolving on nodes that
            // do not have an Avahi-enabled DNS resolver.
	    avahi_address_snprint(address_str, sizeof(address_str), address);
	    toResolve->node->SetHostName(address_str);
	 }
         // Note that Avahi returns the port in host order.
         toResolve->node->SetPort(port);

         // Also, copy the TXT values by iterating through the list of data.
         iterator = txt;
         while (iterator != NULL) {
            // Get data from the TXT record.
            avahi_string_list_get_pair(iterator, &key, &value, &size);
            // Add to the Bonjour record.
            toResolve->node->GetBonjourRecord().AddTXTRecord(key, value);
            // Free data after copy.
            avahi_free(key);
            if (value)
               avahi_free(value);
            // Go to the next TXT record.
            iterator = avahi_string_list_get_next(iterator);
         }

         // Insert now that the node is completely resolved.
         instance->ListOfNodes.push_back(toResolve->node);
         instance->UnLockNodeList();
   }

   // We must free the resolver object since it was created just before calling
   // the resolver procedure.
   avahi_service_resolver_free(r);

   // Also, we should free the resolver data wrapper in order to avoid leaks.
   free(toResolve);

   // Invoke the callback if everything were fine.
   if (event == AVAHI_RESOLVER_FOUND)
      callbackID->callback(callbackID->context);
}

int XrdOucAvahiBonjour::ResolveNodeInformation(XrdOucBonjourResolutionEntry * nodeAndCallback)
{
   // With Avahi, the resolution must be done just after the discovery and
   // on-demand resolution is not supported.
   return 0;
}

/******************************************************************************/
/*        C o n s t r u c t o r s   &   S i n g l e t o n   s t u f f         */
/******************************************************************************/

bool XrdOucAvahiBonjour::XrdOucAvahiBonjourSearchNode::operator()(XrdOucBonjourNode * value)
{
   return strcmp(value->GetBonjourRecord().GetServiceName(), ServiceName) == 0;
}

XrdOucAvahiBonjour * XrdOucAvahiBonjour::_Instance = NULL;

XrdSysMutex XrdOucAvahiBonjour::SingletonMutex;

XrdOucAvahiBonjour::XrdOucAvahiBonjour()
{
   char *env = strdup("AVAHI_COMPAT_NOWARN=1");
   putenv(env);
   poller = avahi_simple_poll_new();
}

XrdOucAvahiBonjour::~XrdOucAvahiBonjour()
{
   if (poller)
      avahi_simple_poll_free(poller);
}

// In this case, to get a portable solution, we are not using any platform
// specific keyword (like volatile on Win), so, to minimize the cost of this
// function (mainly, gaining the lock) is highly recommended that any client of
// this class stores a local reference to the singleton instance in order to
// minimize the number of queries to the lock.
XrdOucAvahiBonjour &XrdOucAvahiBonjour::getInstance()
{
   // At the moment this object is destroyed, the singleton instance will be
   // deleted.
   static XrdOucAvahiBonjourSingletonCleanup cleanGuard;

   SingletonMutex.Lock();
   if (!_Instance)
      _Instance = new XrdOucAvahiBonjour();
   SingletonMutex.UnLock();

   return *_Instance;
}

XrdOucAvahiBonjour::XrdOucAvahiBonjourSingletonCleanup::~XrdOucAvahiBonjourSingletonCleanup()
{
   SingletonMutex.Lock();
   if (_Instance) {
      delete XrdOucAvahiBonjour::_Instance;
      XrdOucAvahiBonjour::_Instance = NULL;
   }
   SingletonMutex.UnLock();
}
