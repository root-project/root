#ifndef __XRDOUCBONJOUR_HH__
#define __XRDOUCBONJOUR_HH__

#include <dns_sd.h>
#include <list>
#include "XrdOuc/XrdOucString.hh"
#include "XrdSys/XrdSysPthread.hh"

#if defined(R__BJRAVAHI)
#include <avahi-client/client.h>
#include <avahi-client/publish.h>
#include <avahi-common/strlst.h>
#endif

class XrdOucBonjourRecord;
class XrdOucBonjourNode;
class XrdOucBonjour;

/******************************************************************************/
/*                      T y p e   d e f i n i t i o n s                       */
/******************************************************************************/

#define TXT_LENGTH 100
#define TIMEOUT 30

#define kBonjourSrvDisabled -1
#define kBonjourSrvBrowse    0
#define kBonjourSrvRegister  1
#define kBonjourSrvBoth      2

// Type definition for the callback function.
typedef void * (*XrdOucBonjourUpdateCallback)(void *);

// Typedef of an struct to store the registration information. Since it is only
// DCO, we do not need to define a class for that.
typedef struct XrdOucBonjourRegisteredEntry {
   XrdOucBonjourRecord * record;
   unsigned short port;
   // Data for the service reference, both Avahi and Bonjour.
#if defined(R__BJRDNSSD)
   // We must mantain this reference alive, since its our link to the mDNS.
   DNSServiceRef bonjourRef;
#elif defined(R__BJRAVAHI)
   // In the case we are using Avahi, lets store the client info structure.
   struct {
      AvahiClient * avahiClient;
      AvahiEntryGroup * avahiEntryGroup;
   } avahiRef;
#endif
} XrdOucBonjourRegisteredEntry;

// Typedef of an struct to store the subcription information. Since it is only
// DCO, we not define a class for that.
typedef struct XrdOucBonjourSubscribedEntry {
   XrdOucBonjourUpdateCallback callback;
   void * context;
   XrdOucString * serviceType;
   // Used to carry the Avahi browser information.
#if defined(R__BJRAVAHI)
   AvahiClient * client;
#endif
} XrdOucBonjourSubscribedEntry;

// Used to make the resolution dinamycally.
typedef struct XrdOucBonjourResolutionEntry {
   XrdOucBonjourNode * node;
   XrdOucBonjourSubscribedEntry * callbackID;
} XrdOucBonjourResolutionEntry;

/******************************************************************************/
/*                        B o n j o u r   r e c o r d                         */
/******************************************************************************/

// Note that this class depends on the compatibility layer of Avahi.
class XrdOucBonjourRecord {
private:
   XrdOucString ServiceName;
   XrdOucString RegisteredType;
   XrdOucString ReplyDomain;
   TXTRecordRef TXTRecord;
   void InitTXTRecord();
   void CopyTXTRecord(const TXTRecordRef &otherRecord);

public:
   XrdOucBonjourRecord() {
      InitTXTRecord();
   }

   XrdOucBonjourRecord(const char * name,
                       const char * type,
                       const char * domain) :
      ServiceName(name), RegisteredType(type), ReplyDomain(domain) {
      InitTXTRecord();
   }

   XrdOucBonjourRecord(const XrdOucBonjourRecord &other) :
      ServiceName(other.ServiceName), RegisteredType(other.RegisteredType),
      ReplyDomain(other.ReplyDomain) {
      InitTXTRecord();
      CopyTXTRecord(other.TXTRecord);
   }

   virtual ~XrdOucBonjourRecord() {
      TXTRecordDeallocate(&TXTRecord);
   }

   const char *GetServiceName() const {
      return ServiceName.length() ? ServiceName.c_str() : NULL;
   }
   const char *GetRegisteredType() const {
      return RegisteredType.length() ? RegisteredType.c_str() : NULL;
   }
   const char *GetReplyDomain() const {
      return ReplyDomain.length() ? ReplyDomain.c_str() : NULL;
   }
   const char *GetTXTRecordData() const {
      return (const char *)TXTRecordGetBytesPtr(&TXTRecord);
   }
   const char *GetTXTValue(const char * key, int &len) const;
   int GetTXTRecordLength() const {
      return TXTRecordGetLength(&TXTRecord);
   }

#if defined(R__BJRAVAHI)
   AvahiStringList *GetTXTAvahiList();
#endif

   int MatchesServiceName(const char * pattern) const {
      return (const_cast<XrdOucString &>(ServiceName)).beginswith(pattern);
   }
   int MatchesRegisteredType(const char * pattern) const {
      return (const_cast<XrdOucString &>(RegisteredType)).beginswith(pattern);
   }
   int MatchesReplyDomain(const char * pattern) const {
      return (const_cast<XrdOucString &>(ReplyDomain)).beginswith(pattern);
   }

   void AddTXTRecord(const char * key, const char * value);
   void AddTXTRecord(const char * key, int value);
   void AddRawTXTRecord(const char * rawData);
   void SetServiceName(const char * name);
   void SetRegisteredType(const char * type);
   void SetReplyDomain(const char * domain);
   void DeleteTXTRecord();

   XrdOucBonjourRecord &operator=(const XrdOucBonjourRecord &other);

   void Print() const;
};

/******************************************************************************/
/*                          B o n j o u r   n o d e                           */
/******************************************************************************/

class XrdOucBonjourNode {
private:
   XrdOucString HostName;
   unsigned short Port;
   XrdOucBonjourRecord BonjourInfo;

public:
   XrdOucBonjourNode() {
      Port = 0;
   }

   XrdOucBonjourNode(const char * hostName,
                     unsigned short port) :
      HostName(hostName) {
      Port = port;
   }

   XrdOucBonjourNode(const char * hostName,
                     unsigned short port,
                     XrdOucBonjourRecord const &bonjourInfo) :
      HostName(hostName), BonjourInfo(bonjourInfo) {
      Port = port;
   }

   XrdOucBonjourNode(XrdOucBonjourRecord const &bonjourInfo) :
      BonjourInfo(bonjourInfo) {
      Port = 0;
   }

   // A handful constructor for the browse reply callback
   XrdOucBonjourNode(const char * name,
                     const char * type,
                     const char * domain) :
      BonjourInfo(name, type, domain) {
      Port = 0;
   }

   XrdOucBonjourNode(const XrdOucBonjourNode &other) :
      HostName(other.HostName), BonjourInfo(other.BonjourInfo) {
      Port = other.Port;
   }

   virtual ~XrdOucBonjourNode() { }

   const char *GetHostName() const {
      return HostName.length() ? HostName.c_str() : NULL;
   }
   unsigned short GetPort() const  {
      return Port;
   }
   const XrdOucBonjourRecord &GetBonjourRecord() const {
      return BonjourInfo;
   }
   XrdOucBonjourRecord &GetBonjourRecord() {
      return BonjourInfo;
   }

   void SetHostName(const char * hostName);
   void SetPort(unsigned short port);
   void SetBonjourRecord(const XrdOucBonjourRecord &record);

   XrdOucBonjourNode &operator=(const XrdOucBonjourNode &other);

   void Print() const;
};

/******************************************************************************/
/*                      B o n j o u r   s e r v i c e s                       */
/******************************************************************************/

class XrdOucBonjour {
protected:
   // List of registered services we have.
   std::list<XrdOucBonjourNode *> ListOfNodes;
   XrdSysMutex ListOfNodesMutex;

public:
   XrdOucBonjour() { }
   virtual ~XrdOucBonjour() { }

   // Register a service on the mDNS local service. This function also
   // subscribes the sender for updates on the discoverage service.
   virtual int RegisterService(XrdOucBonjourRecord &record,
                               unsigned short port = 0) = 0;

   // Subscribes a new client to receive updates about service discoveries.
   // This will detatch a new thread to process the updates, running (when
   // a new update arrives) the callback function in its own thread. This
   // function mush be thread-safe, and its responsability of the client
   // to ensure that.
   virtual int SubscribeForUpdates(const char * servicetype,
                                   XrdOucBonjourUpdateCallback callback,
                                   void * context) = 0;

   // Resolves the name of a node. If you provide a pointer to a node
   // object, this function completes the current information about hostname
   // and port. It is important to use the resolution by-demand since the list
   // may not contain updated information due to the use of highly dynamical
   // DHCP and APIPA addresses.
   virtual int ResolveNodeInformation(XrdOucBonjourResolutionEntry * nodeAndCallback) = 0;

   // Returns the current list of discovered nodes through the Bonjour local
   // mDNS. This list cannot be modified by clients of the class.
   const std::list<XrdOucBonjourNode *> &GetCurrentNodeList() const {
      return ListOfNodes;
   }

   // Methods for locking and unlocking the node table in the case that it
   // will be modified or an exclusive access is needed.
   void LockNodeList() {
      ListOfNodesMutex.Lock();
   }
   void UnLockNodeList() {
      ListOfNodesMutex.UnLock();
   }

   // Accessor to get the singleton instance.
   static XrdOucBonjour &getInstance();
};

/******************************************************************************/
/*                      A b s t r a c t   f a c t o r y                       */
/******************************************************************************/

class XrdOucBonjourFactory {
public:
   static XrdOucBonjourFactory *FactoryByPlatform();

   virtual XrdOucBonjour &GetBonjourManager() = 0;
   virtual ~XrdOucBonjourFactory() { }
};

#endif
