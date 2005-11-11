// @(#)root/reflex:$Name:  $:$Id: Any.h,v 1.3 2005/11/03 15:24:40 roiser Exp $
// Author: Stefan Roiser 2004

// See http://www.boost.org/libs/any for Documentation.

// Copyright Kevlin Henney, 2000, 2001, 2002. All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Any
#define ROOT_Reflex_Any

// What:  variant At boost::any
// who:   contributed by Kevlin Henney,
//        with features contributed and bugs found by
//        Ed Brey, Mark Rodgers, Peter Dimov, and James Curran
// when:  July 2001
// where: tested with BCC 5.5, MSVC 6.0, and g++ 2.95

#include "Reflex/Kernel.h"
#include <algorithm>
#include <typeinfo>
#include <iostream>

namespace ROOT {
  namespace Reflex {
    
    /** 
     * @class Any Any.h Reflex/Any.h
     * @author K. Henney
     */
    class Any {

      friend std::ostream& operator << ( std::ostream&, 
                                         const Any& );

    public: 
      
      /** Constructor */
      Any() 
        : content( 0 ) {}
      
      /** Constructor */
      template< typename ValueType > Any( const ValueType & value ) 
        : content( new holder<ValueType>( value )) {}
      
      /** Copy Constructor */
      Any(const Any & other)
        : content( other.content ? other.content->Clone() : 0 ) {}
      
      ~Any() {
        delete content;
      }
      
      /** Modifier */
      Any & Swap( Any & rhs ) {
        std::swap( content, rhs.content);
        return *this;
      }
      
      /** Modifier */
      template< typename ValueType > Any & operator=( const ValueType & rhs ) {
        Any( rhs ).Swap( * this );
        return * this;
      }

      /** Modifier */
      Any & operator=( const Any & rhs ) {
        Any( rhs ).Swap( * this );
        return * this;
      }
      
      /** Query */
      bool Empty() const {
        return ! content;
      }
      
      /** Query */
      const std::type_info & TypeInfo() const {
        return content ? content->TypeInfo() : typeid( void );
      }

    private:  // or public: ?
      
      /**
       * @class placeholder BoostAny.h Reflex/BoostAny.h
       * @author K. Henney
       */
      class placeholder {
      public: 
        
        /** Destructor */
        virtual ~placeholder() {}
        
        /** Query */
        virtual const std::type_info & TypeInfo() const = 0;
        
        /** Query */
        virtual placeholder * Clone() const = 0;
        
      };
      
      /**
       * @class holder BoostAny.h Reflex/BoostAny.h
       * @author K. Henney
       */
      template< typename ValueType > class holder : public placeholder {
      public: 
        
        /** Constructor */
        holder( const ValueType & value )
          : held( value ) {}
        
        /** Query */
        virtual const std::type_info & TypeInfo() const {
          return typeid( ValueType );
        }
          
        /** Clone */
        virtual placeholder * Clone() const {
          return new holder( held );
        }
        
        /** representation */
        ValueType held;
        
      };
      
      
      /** representation */
      template< typename ValueType > friend ValueType * any_cast( Any * );
      
      // or  public:  
      
      /** representation */
      placeholder * content;
      
    };
    
    
    /**
     * @class bad_any_cast Any.h Reflex/Any.h
     * @author K. Henney
     */
    class bad_any_cast : public std::bad_cast {
    public:
      
      /** Query */
      virtual const char * What() const throw() {
        return "bad_any_cast: failed conversion using any_cast";
      }
    };
    
    /** throw */
    template < class E > void throw_exception( const E & e ) {
      throw e;
    }
    
    /** value */
    template< typename ValueType > ValueType * any_cast( Any * operand ) {
      return operand && operand->TypeInfo() == typeid( ValueType ) 
        ? & static_cast< Any::holder< ValueType > * >( operand->content )->held : 0;
    }
    
    /** value */
    template< typename ValueType > const ValueType * any_cast( const Any * operand ) {
      return any_cast< ValueType >( const_cast< Any * >( operand ));
    }
    
    /** value */
    template< typename ValueType > ValueType any_cast( const Any & operand ) {
      const ValueType * result = any_cast< ValueType >( & operand );
      if ( ! result ) { throw_exception( bad_any_cast()); }
      return * result;
    }

    /** stream operator */
    std::ostream& operator << ( std::ostream&, 
                                const Any& );

  } // namespace Reflex
} // namespace ROOT

#endif // ROOT_Reflex_Any
