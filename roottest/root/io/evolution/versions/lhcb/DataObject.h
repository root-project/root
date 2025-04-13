// $Header: /local/reps/Gaudi/GaudiKernel/GaudiKernel/DataObject.h,v 1.8 2005/01/19 18:31:15 mato Exp $
#ifndef GAUDIKERNEL_DATAOBJECT_H
#define GAUDIKERNEL_DATAOBJECT_H

// Framework include files
/* #include "GaudiKernel/ClassID.h" */
/* #include "GaudiKernel/StatusCode.h" */

// STL includes
#include <string>
#include <ostream>

// Forward declarations
class IOpaqueAddress {};
class StreamBuffer {};
class LinkManager {};
class IRegistry {};


/** @class DataObject DataObject.h GaudiKernel/DataObject.h

    A DataObject is the base class of any identifyable object
    on any data store.
    The base class supplies the implementation of data streaming.

    @author M.Frank
*/
class DataObject   {
private:
  /// Reference count
  unsigned long       m_refCount;
  /// Version number
  unsigned char       m_version;
  /// Pointer to the Registry Object
  IRegistry*          m_pRegistry;
  /// Store of symbolic links
  LinkManager*        m_pLinkMgr;

public:
  DataObject() : m_refCount(0),m_version(0),m_pRegistry(0),m_pLinkMgr(0)
    {}
  
  unsigned long       GetCount() { return m_refCount; }
  unsigned char       GetVersion() { return m_version; }
  IRegistry*          GetRegistry() { return m_pRegistry; }
  LinkManager*        GetLink() { return m_pLinkMgr; }

};
#endif // GAUDIKERNEL_DATAOBJECT_H
