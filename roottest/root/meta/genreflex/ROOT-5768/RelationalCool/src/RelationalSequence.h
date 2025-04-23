// $Id: RelationalSequence.h,v 1.14 2009-12-16 17:17:38 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALSEQUENCE_H
#define RELATIONALCOOL_RELATIONALSEQUENCE_H

// Include files
#include <string>

// Local include files
#include "RelationalException.h"

namespace cool {

  /** @class RelationalSequence RelationalSequence.h
   *
   *  Generic relational implementation of a COOL 'sequence'.
   *
   *  This is actually a table row providing an increasing integer number
   *  in a sequence with no holes, as well as its last modification date.
   *
   *  Presently each 'sequence' is stored as a separate relational table.
   *
   *  WARNING: this is a private class of the implementation library.
   *  An instance of this class can only be used as long as the
   *  RelationalSequenceMgr instance that created it still exists.
   *
   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-03-10
   */

  class RelationalSequence
  {

    friend class RelationalSequenceMgr;

  public:

    /// Destructor
    virtual ~RelationalSequence();

    /// Returns the name of the sequence
    const std::string& name() const
    {
      return m_name;
    }

    /// Returns the current value of the sequence
    /// Throw an exception if the sequence is not yet initialised
    /// Use a select for update by default (must be called in RW transaction)
    unsigned int currVal( bool forUpdate = true )
    {
      return currValDate( 0, forUpdate ).first;
    }

    /// Increments the sequence and return the new current value
    /// Initialise the sequence if the sequence is not yet initialised
    /// The nSteps>0 option increments the value by n steps for bulk insertions
    unsigned int nextVal( unsigned int nSteps = 1 )
    {
      if ( nSteps == 0 )
        throw RelationalException
          ( "Invalid argument=0 to nextVal()", "RelationalSequence" );
      bool forUpdate = true;
      return currValDate( nSteps, forUpdate ).first;
    }

    /// Returns the date corresponding to the current value of the sequence
    /// Throw an exception if the sequence is not yet initialised
    /// Use a select for update by default (must be called in RW transaction)
    const std::string currDate()
    {
      // In principle this always follows nextVal (to get a recent date)
      bool forUpdate = true; // If it follows nextVal, the row is locked anyway
      return currValDate( 0, forUpdate ).second;
    }

  protected:

    /// Constructor from a sequence name and a RelationalSequenceMgr
    RelationalSequence( const std::string& name,
                        const RelationalSequenceMgr& sequenceMgr );

    /// Get the RelationalSequenceMgr reference
    const RelationalSequenceMgr& sequenceMgr() const { return m_sequenceMgr; }

    /// Returns the current (nSteps=0) or the next (nSteps>0) value and date
    /// An exception is thrown if forUpdate is false and nSteps>0
    const std::pair<unsigned, std::string> currValDate( unsigned nSteps,
                                                        bool forUpdate );

  private:

    /// Standard constructor is private
    RelationalSequence();

    /// Copy constructor is private
    RelationalSequence( const RelationalSequence& rhs );

    /// Assignment operator is private
    RelationalSequence& operator=( const RelationalSequence& rhs );

  private:

    /// The name of the sequence (i.e. the name of the sequence table)
    std::string m_name;

    /// Reference to the parent RelationalSequenceMgr
    const RelationalSequenceMgr& m_sequenceMgr;

    /// The first value of the sequence
    static const unsigned int s_firstValue = 0;

    /// The increment value of the sequence
    static const unsigned int s_increment = 1;

  };

}

#endif // RELATIONALCOOL_RELATIONALSEQUENCE_H
