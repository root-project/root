/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef __CINT__
#include <vector>
#ifndef __hpux
#include <iostream>
using namespace std;
#else
#include <iostream.h>
#endif
#endif

template <class TYPE> class ValueVector;

// forward declaration of explitcit instantiation
template<> class ValueVector<double> ;

template <class TYPE> class ValueVector {
public:
   typedef TYPE value_type ;
   //ValueVector();
private:
   TYPE val;
   
#ifndef __CINT__
   typedef            ::ValueVector<value_type>         self_type ;
   typedef  typename  std::vector<value_type>           collection_type ;
#else
   typedef  ValueVector<value_type>    self_type ;
   typedef  std::vector<value_type>    collection_type ;
#endif
   typedef  typename  collection_type::reference        reference ;
   typedef  typename  collection_type::const_reference  const_reference ;
   typedef  typename  collection_type::pointer          pointer ;
   typedef  typename  collection_type::const_pointer    const_pointer ;
   typedef  typename  collection_type::const_iterator   const_iterator ;

   collection_type _container;
public:

//-----------------------------------------------------------------------------
// Primitive Operations
//-----------------------------------------------------------------------------
    void clear(void)                         { _container.clear() ; }
    void assign(const self_type& rhs)        { _container = rhs._container ; }
    bool compare(const self_type& rhs) const { return(_container == rhs._container) ; }

//-----------------------------------------------------------------------------
// Constructors, Destructor, and Assignment
//-----------------------------------------------------------------------------
    ValueVector(void) : _container()                   { }
    ~ValueVector(void)                                 { clear() ; }
    ValueVector(const self_type& rhs) : _container()   { assign(rhs) ; }
    self_type& operator=(const ValueVector<TYPE>& rhs) { if (this != &rhs)
                                                           { clear() ;
                                                             assign(rhs) ;
                                                           }
                                                         return(*this) ;
                                                       }

//-----------------------------------------------------------------------------
// Comparisons
//-----------------------------------------------------------------------------
    bool operator == (const self_type& rhs) const { return( compare(rhs) ) ; }
    bool operator != (const self_type& rhs) const { return(!compare(rhs) ) ; }

//-----------------------------------------------------------------------------
    void print(std::ostream& os = std::cout) const
           { 
           }

};

template <> 
class ValueVector<double> {
public:
    typedef            double                              value_type ;
    ValueVector() { }
private:
   double val;
};


