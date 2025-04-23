// $Id: HvsTagRecord.h,v 1.13 2009-12-16 17:17:37 avalassi Exp $
#ifndef COOLKERNEL_HVSTAGRECORD_H
#define COOLKERNEL_HVSTAGRECORD_H

// Include files
#include "CoolKernel/Time.h"
#include "IHvsTagRecord.h"

// TEMPORARY - debug unknown exception on Windows
//#include <iostream>

namespace cool {

  /** @class HvsTagRecord HvsTagRecord.h
   *
   *  Concrete implementation of an IHvsTagRecord.
   *
   *  @author Andrea Valassi
   *  @date   2006-03-24
   */

  class HvsTagRecord : public IHvsTagRecord {

  public:

    /// Destructor
    virtual ~HvsTagRecord() {}

    /// System-assigned tag ID.
    UInt32 id() const
    {
      return m_id;
    }

    /// Tag scope: node where the tag is defined.
    UInt32 nodeId() const
    {
      return m_nodeId;
    }

    /// Tag name.
    const std::string& name() const
    {
      return m_name;
    }

    /// Tag lock status.
    HvsTagLock::Status lockStatus() const
    {
      return m_lockStatus;
    }

    /// Tag description.
    const std::string& description() const
    {
      return m_description;
    }

    /// Tag insertion time into the database (creation time).
    const ITime& insertionTime() const
    {
      return m_insertionTime;
    }

    /// Constructor from all data members.
    HvsTagRecord( UInt32 id,
                  UInt32 nodeId,
                  const std::string& name,
                  HvsTagLock::Status lockStatus,
                  const std::string& description,
                  const ITime& insertionTime )
      : m_id( id )
      , m_nodeId( nodeId )
      , m_name( name )
      , m_lockStatus( lockStatus )
      , m_description( description )
      , m_insertionTime( insertionTime )
    {
      // TEMPORARY - debug unknown exception on Windows
      //std::cout << "*** HvsTagRecord - CONSTRUCTOR" << std::endl;
    }

    /// Copy constructor.
    /// AV - Added IHvsTagRecord to avoid gcc344 warning on copy constructor
    /// AV - To be tested: would this solve the unknown Windows exception???
    HvsTagRecord( const HvsTagRecord& rhs )
      : IHvsTagRecord()
      , m_id( rhs.m_id )
      , m_nodeId( rhs.m_nodeId )
      , m_name( rhs.m_name )
      , m_lockStatus( rhs.m_lockStatus )
      , m_description( rhs.m_description )
      , m_insertionTime( rhs.m_insertionTime )
    {
      // TEMPORARY - debug unknown exception on Windows
      //std::cout << "*** HvsTagRecord - COPY CONSTRUCTOR" << std::endl;
    }

  private:

    /// Standard constructor is private
    HvsTagRecord();

    /// Assignment operator is private
    HvsTagRecord& operator=( const HvsTagRecord& rhs );

  private:

    /// System-assigned tag ID.
    UInt32 m_id;

    /// Tag scope: node where the tag is defined.
    UInt32 m_nodeId;

    /// Tag name.
    std::string m_name;

    /// Tag lock status.
    HvsTagLock::Status m_lockStatus;

    /// Tag description.
    std::string m_description;

    /// Insertion time into the database.
    Time m_insertionTime;

  };

}

#endif // COOLKERNEL_HVSTAGRECORD_H
