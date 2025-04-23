// $Id: IRecordSelection.h,v 1.15 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IRECORDSELECTION_H
#define COOLKERNEL_IRECORDSELECTION_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolKernel/Record.h"

namespace cool
{

  //--------------------------------------------------------------------------

  /** @class IRecordSelection IRecordSelection.h
   *
   *  Abstract interface to a data record selection.
   *  Inspired by the HARP IEventSelection interface.
   *
   *  This interface is designed to define a user 'payload query cut' which
   *  can be passed as the last argument to the IFolder::browseObjects method.
   *  Payload query cuts are meant to be applied after a preselection on tag,
   *  channel and IOV range is executed on the server using optimized SQL.
   *  Payload queries thus consist in a scan of individual preselected IOVs,
   *  which can be applied either in the C++ client after downloading all
   *  preselected IOVs, or directly on the database server (e.g. by appending
   *  a final SQL selection fragment to the WHERE clause of an RDBMS query).
   *
   *  The 'describe' method is meant to provide an SQL-like (or ROOT-like)
   *  translation of payload query cuts that can be directly appended to
   *  the database server query. In the relational implementation of COOL,
   *  this approach is followed for all 'trusted' concrete implementations
   *  of IRecordSelection defined in the CoolKernel API.
   *
   *  By default, the 'select' method can also be used to apply payload
   *  query cuts on preselected IOVs in the C++ client. In the relational
   *  implementation of COOL, this is the default approach for executing
   *  arbitrary payload queries from user-defined implementation classes.
   *
   *  @author Andrea Valassi
   *  @date   2008-07-05
   */

  class IRecordSelection
  {

  public:

    virtual ~IRecordSelection() {}

    /// Can the selection be applied to a record with the given specification?
    virtual bool canSelect( const IRecordSpecification& spec ) const = 0;

    /// Apply the selection to the given record.
    virtual bool select( const IRecord& record ) const = 0;

    /// Describe the selection (with or without bind variables)
    //virtual const std::string& describe( bool useBindVar = false ) const = 0;

    /// Get the bind variables for the selection.
    //virtual const IRecord& bindVariables() const = 0;

    /// Clone the record selection (and any objects referenced therein).
    virtual IRecordSelection* clone() const = 0;

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IRecordSelection& operator=( const IRecordSelection& rhs );
#endif

  };

}
#endif // COOLKERNEL_IRECORDSELECTION_H
