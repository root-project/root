//--------------------------------------------------------------------------
#ifndef HEPMC_WEIGHT_CONTAINER_H
#define HEPMC_WEIGHT_CONTAINER_H

//////////////////////////////////////////////////////////////////////////
// Matt.Dobbs@Cern.CH, November 2000, refer to:
// M. Dobbs and J.B. Hansen, "The HepMC C++ Monte Carlo Event Record for
// High Energy Physics", Computer Physics Communications (to be published).
//
// Container for the Weights associated with an event or vertex.
//
// This implementation adds a map-like interface in addition to the 
// vector-like interface.
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <string>
#include <map>

namespace HepMC {

    //! Container for the Weights associated with an event or vertex.

    ///
    /// \class  WeightContainer
    /// This class has both map-like and vector-like functionality.
    /// Named weights are now supported.
    class WeightContainer {
	friend class GenEvent;

    public:
        /// defining the size type used by vector and map
	typedef unsigned long long size_type;
        /// iterator for the weight container
	typedef std::vector<double>::iterator iterator;
        /// const iterator for the weight container
	typedef std::vector<double>::const_iterator const_iterator;
	
        /// default constructor
	explicit WeightContainer( size_type n = 0, double value = 0. );
        /// construct from a vector of weights
	WeightContainer( const std::vector<double>& weights );
        /// copy
	WeightContainer( const WeightContainer& in );
	~WeightContainer();

        /// swap
        void swap( WeightContainer & other);
        /// copy assignment
	WeightContainer& operator=( const WeightContainer& );
        /// alternate assignment using a vector of doubles
	WeightContainer& operator=( const std::vector<double>& in );

        /// print weights
	void          print( std::ostream& ostr = std::cout ) const;
        /// write weights in a readable table
	void          write( std::ostream& ostr = std::cout ) const;

        /// size of weight container
	size_type     size() const;
	/// return true if weight container is empty
	bool          empty() const;
	/// push onto weight container
	void          push_back( const double& );
	/// pop from weight container
	void          pop_back();
	/// clear the weight container
	void          clear();

	/// check to see if a name exists in the map
	bool          has_key( const std::string& s ) const;

        /// access the weight container
	double&       operator[]( size_type n );  // unchecked access
        /// access the weight container
	const double& operator[]( size_type n ) const;
        /// access the weight container
	double&       operator[]( const std::string& s );  // unchecked access
        /// access the weight container
	const double& operator[]( const std::string& s ) const;

        /// equality
	bool operator==( const WeightContainer & ) const;
        /// inequality
	bool operator!=( const WeightContainer & ) const;
 	
	/// returns the first element
	double&       front();
	/// returns the first element
	const double& front() const;   
	/// returns the last element
	double&       back();
	/// returns the last element
	const double& back() const;

	/// begining of the weight container
	iterator            begin();
	/// end of the weight container
	iterator            end();
	/// begining of the weight container
	const_iterator      begin() const;
	/// end of the weight container
	const_iterator      end() const;

    private:
        // for internal use only

        /// maplike iterator for the weight container
	/// for internal use only
	typedef std::map<std::string,size_type>::iterator       map_iterator;
        /// const iterator for the weight container
	/// for internal use only
	typedef std::map<std::string,size_type>::const_iterator const_map_iterator;
	/// begining of the weight container
	/// for internal use only
	map_iterator            map_begin();
	/// end of the weight container
	/// for internal use only
	map_iterator            map_end();
	/// begining of the weight container
	/// for internal use only
	const_map_iterator      map_begin() const;
	/// end of the weight container
	/// for internal use only
	const_map_iterator      map_end() const;
	
	/// used by the constructors to set initial names
	/// for internal use only
	void set_default_names( size_type n );
	
    private:
	std::vector<double>          m_weights;
	std::map<std::string,size_type> m_names;
    };

    ///////////////////////////
    // INLINES               //
    ///////////////////////////

    inline WeightContainer::WeightContainer( const WeightContainer& in )
	: m_weights(in.m_weights), m_names(in.m_names)
    {}

    inline WeightContainer::~WeightContainer() {}

    inline void WeightContainer::swap( WeightContainer & other)
    { 
        m_weights.swap( other.m_weights ); 
        m_names.swap( other.m_names ); 
    }

    inline WeightContainer& WeightContainer::operator=
    ( const WeightContainer& in ) {
        /// best practices implementation
	WeightContainer tmp( in );
	swap( tmp );
	return *this;
    }

    inline WeightContainer& WeightContainer::operator=
    ( const std::vector<double>& in ) {
        /// best practices implementation
	WeightContainer tmp( in );
	swap( tmp );
	return *this;
    }

    inline WeightContainer::size_type WeightContainer::size() const { return m_weights.size(); }

    inline bool WeightContainer::empty() const { return m_weights.empty(); }

    inline void WeightContainer::clear() 
    { 
	m_weights.clear(); 
	m_names.clear(); 
    }

    inline double& WeightContainer::operator[]( size_type n ) 
    { return m_weights[n]; }

    inline const double& WeightContainer::operator[]( size_type n ) const
    { return m_weights[n]; }

    inline double& WeightContainer::front() { return m_weights.front(); }

    inline const double& WeightContainer::front() const 
    { return m_weights.front(); }

    inline double& WeightContainer::back() { return m_weights.back(); }

    inline const double& WeightContainer::back() const 
    { return m_weights.back(); }

    inline WeightContainer::iterator WeightContainer::begin() 
    { return m_weights.begin(); }

    inline WeightContainer::iterator WeightContainer::end() 
    { return m_weights.end(); }

    inline WeightContainer::const_iterator WeightContainer::begin() const 
    { return m_weights.begin(); }

    inline WeightContainer::const_iterator WeightContainer::end() const 
    { return m_weights.end(); }

    inline WeightContainer::map_iterator WeightContainer::map_begin() 
    { return m_names.begin(); }

    inline WeightContainer::map_iterator WeightContainer::map_end() 
    { return m_names.end(); }

    inline WeightContainer::const_map_iterator WeightContainer::map_begin() const 
    { return m_names.begin(); }

    inline WeightContainer::const_map_iterator WeightContainer::map_end() const 
    { return m_names.end(); }

} // HepMC

#endif  // HEPMC_WEIGHT_CONTAINER_H
//--------------------------------------------------------------------------



