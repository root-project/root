#ifndef __XRDOUCAPPLEBONJOUR_HH__
#define __XRDOUCAPPLEBONJOUR_HH__

#include <dns_sd.h>
#include <list>
#include "XrdOuc/XrdOucBonjour.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                      B o n j o u r   s e r v i c e s                       */
/******************************************************************************/

class XrdOucAppleBonjour : public XrdOucBonjour {
private:
   // Singleton instance.
   static XrdOucAppleBonjour *_Instance;

   // Mutex to protect the construction of the instance.
   static XrdSysMutex SingletonMutex;

   // All the constructors are private in order to disable object creation,
   // copy or assignment.
   XrdOucAppleBonjour();
   virtual ~XrdOucAppleBonjour();
   XrdOucAppleBonjour(const XrdOucAppleBonjour &) { }
   XrdOucAppleBonjour &operator=(const XrdOucAppleBonjour &) {
      return *this;
   }

   // This a trick to ensure that the singleton object will be deleted when
   // the class is unloaded at program shutdown.
   friend class XrdOucAppleBonjourSingletonCleanup;
   class XrdOucAppleBonjourSingletonCleanup {
   public:
      ~XrdOucAppleBonjourSingletonCleanup();
   };

   class XrdOucAppleBonjourSearchNode {
   private:
      const char * ServiceName;

   public:
      XrdOucAppleBonjourSearchNode(const char * name) {
         ServiceName = name;
      }
      bool operator()(XrdOucBonjourNode * value);
   };

   // List of registered services we have. We mantain this since the service
   // ref is needed to keep alive in order to matain the registration. In
   // this moment a function to release registrations is not needed since
   // protocols live as much as the whole process.
   std::list<XrdOucBonjourRegisteredEntry *> ListOfRegistrations;

   // Callbacks for the DNS-SD C API
   static void RegisterReply(DNSServiceRef ref,
                             DNSServiceFlags flags,
                             DNSServiceErrorType error,
                             const char * name,
                             const char * regtype,
                             const char * domain,
                             void * context);

   static void BrowseReply(DNSServiceRef ref,
                           DNSServiceFlags flags,
                           uint32_t interfaceIndex,
                           DNSServiceErrorType error,
                           const char * name,
                           const char * regtype,
                           const char * domain,
                           void * context);

   static void ResolveReply(DNSServiceRef ref,
                            DNSServiceFlags flags,
                            uint32_t interfaceIndex,
                            DNSServiceErrorType error,
                            const char * fullname,
                            const char * hostname,
                            uint16_t port,
                            uint16_t txtLen,
                            const unsigned char * txtVal,
                            void * context);

   // Internal thread that updates the node list as updates come from the
   // dns responder.
   pthread_t BrowseEventLoopThreadInfo;
   static void * BrowseEventLoopThread(void * context);

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
   static XrdOucAppleBonjour &getInstance();
};

/******************************************************************************/
/*                      A b s t r a c t   f a c t o r y                       */
/******************************************************************************/

class XrdOucAppleBonjourFactory : public XrdOucBonjourFactory {
   XrdOucBonjour &GetBonjourManager() {
      return XrdOucAppleBonjour::getInstance();
   }
};

#endif
