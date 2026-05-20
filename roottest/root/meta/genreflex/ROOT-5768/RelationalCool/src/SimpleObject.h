// $Id: SimpleObject.h,v 1.12 2009-12-17 18:38:53 avalassi Exp $
#ifndef RELATIONALCOOL_SIMPLEOBJECT_H
#define RELATIONALCOOL_SIMPLEOBJECT_H

// Include files
#include <boost/shared_ptr.hpp>
#include <vector>
#include "CoolKernel/ValidityKey.h"

namespace cool {

  // Forward declarations
  class SimpleObject;
  typedef std::vector<SimpleObject> SOVector;
  typedef std::vector<SimpleObject>::const_iterator SOIterator;
  std::ostream &operator<<( std::ostream& s, const SimpleObject& o );

  /** SimpleObject.h
   *
   * A basic object 'shell' used for lightweight object comparison/handling.
   * It encapsulated the key attributes of an object without payload.
   *
   * @author Sven A. Schmidt and Andrea Valassi
   * @date 2005-02-11
   */

  class SimpleObject
  {
  public:

    unsigned int objectId;
    ChannelId channelId;
    ValidityKey since;
    ValidityKey until;

    SimpleObject( unsigned int anObjectId,
                  const ChannelId& aChannelId,
                  const ValidityKey& aSince,
                  const ValidityKey& anUntil )
    {
      objectId = anObjectId;
      channelId = aChannelId;
      since = aSince;
      until = anUntil;
    }

    /// Only consider the object id in comparison
    bool operator==( const SimpleObject& rhs ) const
    {
      return objectId == rhs.objectId;
    }

    SOVector intersect( const SOVector& objects ) const
    {
      SOVector res;
      for ( SOIterator obj = objects.begin(); obj != objects.end(); ++obj ) {
        if ( overlaps( *obj ) ) res.push_back( *obj );
      }
      return res;
    }

    bool overlaps( const SimpleObject& obj ) const
    {
      if ( channelId != obj.channelId ) {
        return false;
      } else {
        return ( since <= obj.since && obj.since < until )
          || ( obj.since <= since && since < obj.until );
      }
    }

    SOVector filter( const SimpleObject& obj ) const
    {
      SOVector res;
      if ( ! overlaps( obj ) ) {
        res.push_back( obj );
        return res;
      }
      if ( obj.since < since )
        res.push_back
          ( SimpleObject( obj.objectId, obj.channelId, obj.since, since ) );
      if ( obj.until > until )
        res.push_back
          ( SimpleObject( obj.objectId, obj.channelId, until, obj.until ) );
      return res;
    }

    SOVector visibleThrough( const SOVector& objects ) const
    {
      SOVector res( 1, *this );
      SOVector tmp1, tmp2;
      for ( SOIterator obj = objects.begin(); obj != objects.end(); ++obj ) {
        tmp1.clear();
        for ( SOIterator source = res.begin();
              source != res.end();
              ++source ) {
          tmp2 = obj->filter( *source );
          tmp1.insert( tmp1.end(), tmp2.begin(), tmp2.end() );
        }
        res.clear();
        res.insert( res.end(), tmp1.begin(), tmp1.end() );
      }
      return res;
    }

  };

  /// Streamer for SimpleObject objects
  inline std::ostream &operator<<( std::ostream& s, const SimpleObject& o )
  {
    s << o.objectId << ", " << o.channelId
      << " [" << o.since << "," << o.until << "]";
    return s;
  }

  /// Less than comparison functor to compare SimpleObject since
  struct lt_since
    : public std::binary_function<SimpleObject, SimpleObject, bool>
  {
    bool operator()( const SimpleObject& lhs,
                     const SimpleObject& rhs ) const {
      return ( lhs.since < rhs.since  );
    }
  };

} // namespace

#endif
