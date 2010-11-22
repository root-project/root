#ifndef __XRDOUCAvahiBONJOUR_HH__
#define __XRDOUCAvahiBONJOUR_HH__

#include <avahi-client/client.h>
#include <avahi-client/lookup.h>
#include <avahi-client/publish.h>
#include <avahi-common/alternative.h>
#include <avahi-common/simple-watch.h>
#include <avahi-common/malloc.h>
#include <avahi-common/error.h>
#include <avahi-common/timeval.h>

#include <list>
#include "XrdOuc/XrdOucBonjour.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                      B o n j o u r   s e r v i c e s                       */
/******************************************************************************/

class XrdOucAvahiBonjour : public XrdOucBonjour {
private:
   // Singleton instance.
   static XrdOucAvahiBonjour *_Instance;

   // Mutex to protect the construction of the instance.
   static XrdSysMutex SingletonMutex;

   // All the constructors are private in order to disable object creation,
   // copy or assignment.
   XrdOucAvahiBonjour();
   virtual ~XrdOucAvahiBonjour();
   XrdOucAvahiBonjour(const XrdOucAvahiBonjour &) { }
   XrdOucAvahiBonjour &operator=(const XrdOucAvahiBonjour &) {
      return *this;
   }

   // This a trick to ensure that the singleton object will be deleted when
   // the class is unloaded at program shutdown.
   friend class XrdOucAvahiBonjourSingletonCleanup;
   class XrdOucAvahiBonjourSingletonCleanup {
   public:
      ~XrdOucAvahiBonjourSingletonCleanup();
   };

   class XrdOucAvahiBonjourSearchNode {
   private:
      const char * ServiceName;

   public:
      XrdOucAvahiBonjourSearchNode(const char * name) {
         ServiceName = name;
      }
      bool operator()(XrdOucBonjourNode * value);
   };

   // List of registered services we have. We mantain this since the service
   // ref is needed to keep alive in order to matain the registration. In
   // this moment a function to release registrations is not needed since
   // protocols live as much as the whole process.
   std::list<XrdOucBonjourRegisteredEntry *> ListOfRegistrations;

   // Internal thread that updates the node list as updates come from the
   // dns responder.
   pthread_t BrowseEventLoopThreadInfo;
   static void * BrowseEventLoopThread(void * context);

   // Avahi poller that implements the event loop.
   AvahiSimplePoll * poller;

   // Callback functions for the Avahi services (similar to the ones in the
   // Apple Bonjour version).
   static void RegisterReply(AvahiClient *c,
                             AvahiClientState state,
                             void * userdata);

   static void RegisterEntries(XrdOucBonjourRegisteredEntry * entry);

   static void EntryGroupReply(AvahiEntryGroup *g,
                               AvahiEntryGroupState state,
                               void *userdata);

   static void ClientReply(AvahiClient *c,
                           AvahiClientState state,
                           void * userdata);

   static void BrowseReply(AvahiServiceBrowser *b,
                           AvahiIfIndex interface,
                           AvahiProtocol protocol,
                           AvahiBrowserEvent event,
                           const char *name,
                           const char *type,
                           const char *domain,
                           AvahiLookupResultFlags flags,
                           void* userdata);

   static void ResolveReply(AvahiServiceResolver *r,
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
                            void* userdata);

public:
   // Register a service on the mDNS local service. This funcion also
   // subscribes the sender for updates on the discoverage service.
   int RegisterService(XrdOucBonjourRecord &record,
                       unsigned short port = 0);

   // Subscribes a new client to receive updates about service discoveries.
   // This will detatch a new thread to process the updates, running (when
   // a new update arrives) the callback function in its own thread. This
   // function mush be thread-safe, and its responsability of the client
   // to ensure that.
   int SubscribeForUpdates(const char * servicetype,
                           XrdOucBonjourUpdateCallback callback,
                           void * context);

   // Resolves the name of a node. If you provides a pointer to a node
   // object, this function completes the current information about hostname
   // and port. It is important to use the resolution by-demand since the list
   // may not contain updated information due to the use of highly dynamical
   // DHCP and APIPA addresses.
   int ResolveNodeInformation(XrdOucBonjourResolutionEntry * nodeAndCallback);

   // Accessor to get the singleton instance.
   static XrdOucAvahiBonjour &getInstance();
};

/******************************************************************************/
/*                      A b s t r a c t   f a c t o r y                       */
/******************************************************************************/

class XrdOucAvahiBonjourFactory : public XrdOucBonjourFactory {
   XrdOucBonjour &GetBonjourManager() {
      return XrdOucAvahiBonjour::getInstance();
   }
};

#endif
