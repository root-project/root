/*
   C++ implementation of Bonjour services.  Code is based
   on the implementation of TBonjour* classes written by
   Fons Rademakers for the ROOT Framework.
*/

#include <arpa/inet.h>
#include <net/if.h>
#include <sys/select.h>
#include <cstdlib>
#include "XrdOuc/XrdOucBonjour.hh"
#include "XrdSys/XrdSysError.hh"
#include "Xrd/XrdProtLoad.hh"

// Conditional inclusion of headers dependent on platform.
#include "XrdOuc/XrdOucFactoryBonjour.hh"

/******************************************************************************/
/*                      G l o b a l   V a r i a b l e s                       */
/******************************************************************************/

extern XrdSysError  XrdLog;               // Defined in XrdMain.cc

/******************************************************************************/
/*                        B o n j o u r   r e c o r d                         */
/******************************************************************************/

#if !defined(__macos__)
AvahiStringList *XrdOucBonjourRecord::GetTXTAvahiList()
{
   AvahiStringList *list;
   uint16_t len, i;
   char key[256];
   const void *rawData, *value;
   uint8_t valueLen;
   XrdOucString entry;
   char valueTrimmed[1024];

   // Initialize common data.
   list = NULL;
   i = 0;
   key[0] = '\0';
   value = NULL;

   // Get TXT record raw data.
   rawData = TXTRecordGetBytesPtr(&TXTRecord);
   len = TXTRecordGetLength(&TXTRecord);

   // Iterate through all the elements of the TXT record list creating a list
   // item for all of them.
   while (TXTRecordGetItemAtIndex(len, rawData, i, 256, key, &valueLen, &value) != kDNSServiceErr_Invalid) {
      // Empty string.
      entry.hardreset();
      // Construct the entry according to the mDNS TXT management rules.
      entry.append(key);
      if (value != NULL) {
         entry.append("=");
         if (valueLen > 0) {
            // The 'n' version of strcpy is more secure.
            strncpy(valueTrimmed, (const char *)value, valueLen);
            // Terminate the string to ensure buffer security.
            valueTrimmed[valueLen] = '\0';
            // Now, append.
            entry.append(valueTrimmed);
         }
      }
      // Build the Avahi List Entry. Note that you must free this list after
      // using it with the avahi_string_list_free() function.
      list = avahi_string_list_add(list, entry.c_str());
      i++;
   }

   // Return the list.
   return list;
}
#endif

void XrdOucBonjourRecord::AddTXTRecord(const char * key, const char * value)
{
   TXTRecordSetValue(&TXTRecord, key, strlen(value), value);
}

void XrdOucBonjourRecord::AddTXTRecord(const char * key, int value)
{
   char value_str[256];

   snprintf(value_str, 256, "%d", value);

   AddTXTRecord(key, value_str);
}

void XrdOucBonjourRecord::SetServiceName(const char * name)
{
   // This method is specially maded for use when a local installation of
   // multiple processes listening on different ports. It is more efficient
   // to do an assign than to create a new object, furthermore, if the object
   // is on the stack, this method minimizes the use of the heap (at least
   // directly, since XrdOucString makes use of it for character storage).
   ServiceName.assign(name, 0);
}

void XrdOucBonjourRecord::SetRegisteredType(const char * type)
{
   RegisteredType.assign(type, 0);
}

void XrdOucBonjourRecord::SetReplyDomain(const char * domain)
{
   ReplyDomain.assign(domain, 0);
}

void XrdOucBonjourRecord::DeleteTXTRecord()
{
   TXTRecordDeallocate(&TXTRecord);
   InitTXTRecord();
}

XrdOucBonjourRecord & XrdOucBonjourRecord::operator=(const XrdOucBonjourRecord &other)
{
   if (this != &other) {
      ServiceName.assign(other.ServiceName, 0);
      RegisteredType.assign(other.RegisteredType, 0);
      ReplyDomain.assign(other.ReplyDomain, 0);
      TXTRecordDeallocate(&TXTRecord);
      InitTXTRecord();
      CopyTXTRecord(other.TXTRecord);
   }

   return *this;
}

void XrdOucBonjourRecord::Print() const
{
   XrdLog.Say("INFO: Bonjour RECORD = ", GetServiceName(), GetRegisteredType(), GetReplyDomain());
   XrdLog.Say("INFO: Bonjour TXT = ", GetTXTRecordData());
}

void XrdOucBonjourRecord::AddRawTXTRecord(const char * rawData)
{
   uint16_t i = 0, len;
   char key[256];
   uint8_t valueLen;
   const void * value;

   TXTRecordDeallocate(&TXTRecord);
   InitTXTRecord();

   len = strlen(rawData);
   while (TXTRecordGetItemAtIndex(len, rawData, i, 256, key, &valueLen, &value) != kDNSServiceErr_Invalid) {
      TXTRecordSetValue(&TXTRecord, key, valueLen, value);
      i++;
   }
}

void XrdOucBonjourRecord::InitTXTRecord()
{
   TXTRecordCreate(&TXTRecord, 0, NULL);
}

void XrdOucBonjourRecord::CopyTXTRecord(const TXTRecordRef &otherRecord)
{
   uint16_t i = 0, len;
   const void * rawTXT;
   char key[256];
   uint8_t valueLen;
   const void * value;

   len = TXTRecordGetLength(&otherRecord);
   rawTXT = TXTRecordGetBytesPtr(&otherRecord);
   while (TXTRecordGetItemAtIndex(len, rawTXT, i, 256, key, &valueLen, &value) != kDNSServiceErr_Invalid) {
      TXTRecordSetValue(&TXTRecord, key, valueLen, value);
      i++;
   }
}

const char * XrdOucBonjourRecord::GetTXTValue(const char * key, int &valueLen) const
{
   uint16_t len;
   uint8_t valLen;
   const void * rawTXT;

   len = TXTRecordGetLength(&TXTRecord);
   rawTXT = TXTRecordGetBytesPtr(&TXTRecord);

   // Copy the results.
   rawTXT = TXTRecordGetValuePtr(len, rawTXT, key, &valLen);
   valueLen = valLen;

   return (const char *)rawTXT;
}

/******************************************************************************/
/*                          B o n j o u r   n o d e                           */
/******************************************************************************/

void XrdOucBonjourNode::SetHostName(const char * hostName)
{
   HostName.assign(hostName, 0);
}

void XrdOucBonjourNode::SetPort(unsigned short port)
{
   Port = port;
}

void XrdOucBonjourNode::SetBonjourRecord(const XrdOucBonjourRecord &record)
{
   BonjourInfo = record;
}

XrdOucBonjourNode & XrdOucBonjourNode::operator=(const XrdOucBonjourNode &other)
{
   if (this != &other) {
      HostName.assign(other.HostName, 0);
      Port = other.Port;
      BonjourInfo = other.BonjourInfo;
   }

   return *this;
}

void XrdOucBonjourNode::Print() const
{
   char port[36];
   snprintf(port, 36, "%d (%p)", GetPort(), this);
   const char *host = GetHostName() ? GetHostName() : "<empty>";
   XrdLog.Say("INFO: Bonjour NODE = ", host, ":", port);
   GetBonjourRecord().Print();
}

/******************************************************************************/
/*                      A b s t r a c t   f a c t o r y                       */
/******************************************************************************/

XrdOucBonjourFactory *XrdOucBonjourFactory::FactoryByPlatform()
{
   // Construct a factory object depending on the plaform we are running. This
   // is resolved at compilation time, so it is fast, but static.
   // There is room to improvement here, refining how we detect the operating
   // system and taking into account how to deal with Windows, since there is a
   // project to port Avahi to Win32 (natively) and mDNS is currently supported.
#if defined(__macos__)  // We are on Mac OS X, so load the mDNS version.
   return new XrdOucAppleBonjourFactory();
#elif defined(__linux__) // We are on GNU/Linux, so load the Avahi version.
   return new XrdOucAvahiBonjourFactory();
#else // Currently, Windows is not supported (altough with mDNS can be).
   return NULL;
#endif
}
