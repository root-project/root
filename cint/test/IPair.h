/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef _IPair_
#define _IPair_

#include <iostream>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdlib>

/*!
ICoordinate provides an abstract integer type for IPair.  Currently
ICoordinate is defined to be int.  Using ICoordinate instead of int
to represent a single position coordinate will allow for seamless
forward compatibility should the underlying position representation
in  change.
*/

typedef int ICoordinate;

/*!
IPair provides a 2D Vector class with integer components.  IPair
may be used to represent 2D positon when interpreted as a
displacement from the origin.  Supports general vector and
component-wise mathematics.  All member functions are non-modifying
(with the exception of assignment operators =, +=, *=, etc. ).  Some
vector functions cannot be implemented well in integer space, such
as distance (which involves square roots) or products which are
subject to integer overflow; see DPair for a double precision 2D
Vector class.
*/

class IPair
{

public:

	// Object construction
	IPair();
	IPair( ICoordinate, ICoordinate );
        IPair( const IPair & );

	IPair &operator += ( const IPair & );
	IPair &operator -= ( const IPair & );
	IPair &operator *= ( const IPair & );
	IPair &operator /= ( const IPair & );
	
	IPair operator + () const;
	IPair operator - () const;

	IPair operator * ( ICoordinate ) const;
	IPair operator / ( ICoordinate ) const;
	IPair &operator *= ( ICoordinate );
	IPair &operator /= ( ICoordinate );

	// Element access
	ICoordinate x() const;
	ICoordinate y() const;
	ICoordinate operator [] ( unsigned ) const;
	ICoordinate &y();
	ICoordinate &x();
	ICoordinate &operator [] ( unsigned );

	// Scalar operations
	ICoordinate Min() const;
	ICoordinate Max() const;
	ICoordinate AbsMax() const;
	ICoordinate AbsMin() const;
	static ICoordinate Dot( const IPair &a, const IPair &b );
	static ICoordinate Signum( ICoordinate );
	double DistanceTo( const IPair & ) const;
	double Mag() const;
	double MagSq() const;

	// Vector operations
	IPair Abs() const;
	IPair Signum() const;
	IPair Perpendicular() const;
	static ICoordinate Cross( const IPair &a, const IPair &b );
	static IPair Min( const IPair &a, const IPair &b );
	static IPair Max( const IPair &a, const IPair &b );
	IPair RotateCW() const;
	IPair RotateCCW() const;
	IPair Rotate180() const;
	IPair RotateCW( const IPair &origin ) const;
	IPair RotateCCW( const IPair &origin ) const;
	IPair Rotate180( const IPair &origin ) const;

	// Constants
	static IPair MostPositive();
	static IPair MostNegative();

	// Testing
	static bool IsManhattan( const IPair &a, const IPair &b );

	// Useful for STL
	// Treated as x, then y
	bool operator <  ( const IPair & ) const;
	bool operator >  ( const IPair & ) const;
	bool operator <= ( const IPair & ) const;
	bool operator >= ( const IPair & ) const;
	bool operator == ( const IPair & ) const;
	bool operator != ( const IPair & ) const;

private:

	friend IPair operator + ( const IPair &, const IPair & );
	friend IPair operator - ( const IPair &, const IPair & );
	friend IPair operator * ( const IPair &, const IPair & );
	friend IPair operator / ( const IPair &, const IPair & );

	ICoordinate v[2];
};

std::ostream &operator << ( std::ostream &, const IPair & );

// Arithmetic (vector) operations...
IPair operator + ( const IPair &, const IPair & );
IPair operator - ( const IPair &, const IPair & );
IPair operator * ( const IPair &, const IPair & );
IPair operator / ( const IPair &, const IPair & );

IPair operator * ( ICoordinate, const IPair & );

// Avoid CINT double-definition
#ifndef __RUNTIME_ONLY__

//
// Inline function definitions follow below
//


// Object construction

/*!
Default constructor for a IPair.  Both coordinates are initialized to 0.
*/
inline IPair::IPair()
{
	v[0] = v[1] = 0;
}

/*!
Constructor for a IPair.
\param x Initial x coordinate
\param y Initial y coordinate
*/
inline IPair::IPair( ICoordinate x, ICoordinate y )
{
	v[0] = x;
	v[1] = y;
}

/*!
Copy constructor for a IPair.
\param p Initial values
*/

inline IPair::IPair( const IPair &p ) 
{
	v[0] = p.v[0];
        v[1] = p.v[1];
}

// Arithmetic (vector) operations

/*!
\param V The addend of a addition.
\return The component-wise addition of two vectors.
*/
//
inline IPair operator + ( const IPair &U, const IPair &V )
{
	return IPair( U.v[0] + V.v[0], U.v[1] + V.v[1] );
}

/*!
\param V The subtrahend of a subtraction.
\return The component-wise difference of two vectors.
*/
//
inline IPair operator - ( const IPair &U, const IPair &V )
{
	return IPair( U.v[0] - V.v[0], U.v[1] - V.v[1] );
}

/*!
\param V The multiplier of a multiplication.
\return The component-wise multiplication of two vectors.
*/
//
inline IPair operator * ( const IPair &U, const IPair &V )
{
	return IPair( U.v[0] * V.v[0], U.v[1] * V.v[1] );
}

/*!
\param V The divisor of a division.
\return The component-wise division of two vectors.
*/
//
inline IPair operator / ( const IPair &U, const IPair &V )
{
	return IPair( U.v[0] / V.v[0], U.v[1] / V.v[1] );
}


/*!
\param V The addend of addition.
\return The component-wise assignment addition of two vectors.
*/
inline IPair &IPair::operator += ( const IPair &V )
{
	v[0] += V.v[0];
	v[1] += V.v[1];

	return( * this );
}

/*!
\param V The subtrahend of a subtraction.
\return The vector component-wise assignment difference of two vectors.
*/
inline IPair &IPair::operator -= ( const IPair &V )
{
	v[0] -= V.v[0];
	v[1] -= V.v[1];

	return( * this );
}

/*!
\param V The multiplier of a multiplication.
\return The component-wise assignment multiplication of two vectors
*/
inline IPair &IPair::operator *= ( const IPair &V )
{
	v[0] *= V.v[0];
	v[1] *= V.v[1];

	return( * this );
}

/*!
\param V The divisor of a division.
\return The component-wise assignment division of two vectors
*/
inline IPair &IPair::operator /= ( const IPair &V )
{
	v[0] /= V.v[0];
	v[1] /= V.v[1];

	return( * this );
}

/*!
Unary plus... does nothing! (For symmetry with unary - only)
\return this (unchanged)
*/
inline IPair IPair::operator + () const
{
	return *this;
}

/*!
Unary minus, negates a vector.
\return (component-wise) negation of the current IPair.
*/
inline IPair IPair::operator - () const
{
	return IPair( -v[0], -v[1] );
}

/*!
\param i Scalar multiplier.
\return The component-wise multiplication of a vector by a scalar value
*/
inline IPair IPair::operator * ( ICoordinate i ) const
{
	return IPair( v[0] * i, v[1] * i );
}

/*!
\param i Scalar divisor
\return The component-wise division of a vector by a scalar value
*/
inline IPair IPair::operator / ( ICoordinate i ) const
{
	return IPair( v[0] / i, v[1] / i );
}

/*!
\param i A scalar multiplier;
\return The component-wise assignment multiplication of a vector by a scalar value
*/
inline IPair &IPair::operator *= ( ICoordinate i )
{
	v[0] *= i;
	v[1] *= i;

	return ( *this );
}

/*!
\param i The divisor of a division.
\return The component-wise assignment division of two vectors
*/
inline IPair &IPair::operator /= ( ICoordinate i )
{
	v[0] /= i;
	v[1] /= i;

	return ( *this );
}


// Element access

/*!
\return The x (0th) component of the vector.
*/
inline ICoordinate IPair::x() const
{
	return v[0];
}

/*!
\return The y (1st) component of the vector.
*/
inline ICoordinate IPair::y() const
{
	return v[1];
}

/*!
\param u Component to return (must be 0 or 1)
\return The u'th component
*/
inline ICoordinate IPair::operator [] ( unsigned u ) const
{
	return v[u];
}

/*!
\return A reference to the x (0th) component of the vector
*/
inline ICoordinate &IPair::x()
{
	return v[0];
}

/*!
\return A reference to the y (1st) component of the vector
*/
inline ICoordinate &IPair::y()
{
	return v[1];
}

/*!
\param u Component to return (must be 0 or 1)
\return A reference to the u'th component
*/
inline ICoordinate &IPair::operator [] ( unsigned u )
{
	return v[u];
}


// Scalar operations

/*!
\return The minimum of the x and y components of the vector
*/
inline ICoordinate IPair::Min() const
{
	return v[0] < v[1] ? v[0] : v[1];
}

/*!
\return The maximum of the x and y components of the vector
*/
inline ICoordinate IPair::Max() const
{
	return v[0] > v[1] ? v[0] : v[1];
}

/*!
\return The component-wise absolute value of this IPair.
*/
inline IPair IPair::Abs() const
{
	return IPair( std::abs( v[0] ), std::abs( v[1] ) );
}

/*!
\return The minimum of the absolute value of the x and y components of the vector
*/
inline ICoordinate IPair::AbsMin() const
{
	return Abs().Min();
}

/*!
\return The maximum of the absolute value of the x and y components of the vector
*/
inline ICoordinate IPair::AbsMax() const
{
	return Abs().Max();
}

/*!
Inner product... be careful, it is quite common for the product of two integer vaules to overflow integer space.  It is recommended that DPair conversion be used for this operation.
\param a A IPair
\param b Another IPair
\return The vector dot (inner) product of two vectors
*/
inline ICoordinate IPair::Dot( const IPair &a, const IPair &b )
{
	return( a[0] * b[0] + a[1] * b[1] );
}

/*!
\param i The value to test the signum (sign) of
\return -1,0,1 if the value of i is <0, ==0, or >0 respectively.
*/
inline ICoordinate IPair::Signum( ICoordinate i )
{
	if ( i > 0 ) return 1;
	if ( i < 0 ) return -1;
	return 0;
}


/*!
\return The square of the Euclidean norm of this IPair (See Mag()).
*/
inline double IPair::MagSq() const
{
	double a = v[0];
	double b = v[1];
	
	return a*a + b*b;
}

/*!
\return The Euclidean norm (distance from the origin to the tip) of this IPair.
*/
inline double IPair::Mag() const
{
	if ( v[0] == 0 ) return std::abs( v[1] );
	if ( v[1] == 0 ) return std::abs( v[0] );
	
	return sqrt( MagSq() );
}

/*!
\return Euclidean distance from this point to another point
*/
inline double IPair::DistanceTo( const IPair &p ) const
{
	return ( *this - p ).Mag();

// 	double dx = v[0] - p.v[0];
// 	double dy = v[1] - p.v[1];

// 	return sqrt( dx*dx + dy*dy );
}

// Vector operations

/*!
\return A signum vector, the integer equivlent of a unit
vector.  The vector returned has components of -1,0,1 depending if
the original vector components were negative, zero, or positive,
respectively.  May NOT have unit magnitude.  For a true unit vector,
consider converting to a DPair first.
*/
inline IPair IPair::Signum() const
{
	return IPair( Signum( v[0] ), Signum( v[1] ) );
}

/*!
\return A vector perpendicular to this one, counter-clockwise, of the same magnitude of this IPair.
*/
inline IPair IPair::Perpendicular() const
{
	return IPair( -v[1], v[0] );
}

/*!
Warning, integer products may result in an overflow during calculation.  Consider converting to a DPair first.
\param a The first IPair
\param b The second IPair
\return The scalar component of the vector cross product of two IPairs.
*/
inline ICoordinate IPair::Cross( const IPair &a, const IPair &b )
{
	return a.v[0]*b.v[1] - a.v[1]*b.v[0];
}

/*!
\param a A IPair
\param b Another IPair
\return The component-wise minimum of two vectors
*/
inline IPair IPair::Min( const IPair &a, const IPair &b )
{
	return IPair(
		a.v[0] < b.v[0] ? a.v[0] : b.v[0],
		a.v[1] < b.v[1] ? a.v[1] : b.v[1]
		);
}
		
/*!
\param a A IPair
\param b Another IPair
\return The component-wise maximum of two vectors
*/
inline IPair IPair::Max( const IPair &a, const IPair &b )
{
	return IPair(
		a.v[0] > b.v[0] ? a.v[0] : b.v[0],
		a.v[1] > b.v[1] ? a.v[1] : b.v[1]
		);
}

/*!
\return A copy of this IPair rotated 90 degrees counter-clockwise about the origin.
*/
inline IPair IPair::RotateCCW() const
{
	return IPair( -v[1], v[0] );
}

/*!
\return A copy of this IPair rotated 90 degrees counter-clockwise about the given point.
\param origin Point to rotate about
*/
inline IPair IPair::RotateCCW( const IPair &origin ) const
{
	return origin + ( *this - origin ).RotateCCW();
}

/*!
\return A copy of this IPair rotated 90 degrees clockwise about the origin.
*/
inline IPair IPair::RotateCW() const
{
	return IPair( v[1], -v[0] );
}

/*!
\return A copy of this IPair rotated 90 degrees clockwise about the given point.
\param origin Point to rotate about
*/
inline IPair IPair::RotateCW( const IPair &origin ) const
{
	return origin + ( *this - origin ).RotateCW();
}

/*!
\return A copy of this IPair rotated 180 degrees about the origin.
*/
inline IPair IPair::Rotate180() const
{
	return IPair( -v[0], -v[1] );
}

/*!
\return A copy of this IPair rotated 180 degrees about the given point.
\param origin Point to rotate about
*/
inline IPair IPair::Rotate180( const IPair &origin ) const
{
	return origin - *this;
}

/*!
\return The IPair with the largest representable x and y components
*/
inline IPair IPair::MostPositive()
{
	return IPair( INT_MAX, INT_MAX );
}

/*!
\return The IPair with the smallest (largest negative) representable x and y components
*/
inline IPair IPair::MostNegative()
{
	return IPair( INT_MIN, INT_MIN );
}

// Testing
/*!
\return True if either the x or y (or both) components of both IPairs are equal.
*/
inline bool IPair::IsManhattan( const IPair &a, const IPair &b )
{
	return a[0] == b[0] || a[1] == b[1];
}

/*!
Tests if one IPair is less than another with precedence X,Y
\return True if the X component of this IPair is less than the other IPair's X component, or in the case that the X components are equal, True if the Y component is less than the other's.  False otherwise.
\param V IPair to compare to.
*/
inline bool IPair::operator < ( const IPair &V ) const
{
	// Define an ordering with precedence X,Y
	if ( v[0] < V.v[0] ) return ( true );
	if ( v[0] > V.v[0] ) return ( false );
	
	// Y's are equal	
	if ( v[1] < V.v[1] ) return ( true );
//	if ( v[1] > V.v[1] ) return ( false ); redundant
	
	// X's are equal	
	return ( false );
}

/*!
Tests if one IPair is greater than another with precedence X,Y
\return True if the X component of this IPair is greater than the other IPair's X component, or in the case that the X components are equal, True if the Y component is greater than the other's.  False otherwise.
\param V IPair to compare to.
*/
inline bool IPair::operator > ( const IPair &V ) const
{
	// Pass the job off to <
	return ( V < *this );
}

/*!
\return True only if all components of both IPairs match, false otherwise.
\param V IPair to compare to.
*/
inline bool IPair::operator == ( const IPair &V ) const
{
	return ( v[0] == V.v[0] && v[1] == V.v[1] );
}

/*!
\return True if at least one component of the two IPairs do not match, false all match.
\param V IPair to compare to.
*/
inline bool IPair::operator != ( const IPair &V ) const
{
	// Let == do the work
	return ( ! ( *this == V ) );
}

/*!
\return True if this IPair is greater than or equal to another (see <)
\param V IPair to compare to.
*/
inline bool IPair::operator >= ( const IPair &V ) const
{
	return ( *this == V || *this > V );
}

/*!
\return True if this IPair is less than or equal to another (see <)
\param V IPair to compare to.
*/
inline bool IPair::operator <= ( const IPair &V ) const
{
	return ( *this == V || *this < V );
}


/*!
   Double pre-multiply
	\param V IPair to be scaled
	\param i Scaling amount
	\return IPair V scaled by i
*/
inline IPair operator * ( ICoordinate i, const IPair &V )
{
	return IPair( V[0] * i, V[1] * i );
}

#endif // __RUNTIME_ONLY__

#endif // _IPair_

