/*
  File: roottest/python/ttree/TTreeTypes.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 11/17/06
  Last: 04/22/08
*/

#include <vector>


#ifdef __MAKECINT__
#pragma link C++ class std::vector< vector< float > >;
#pragma link C++ class std::vector< vector< float > >::iterator;
#pragma link C++ class std::vector< vector< float > >::const_iterator;
#endif


typedef std::vector< float >                 Floats_t;
typedef std::vector< std::vector< float > >  Tuples_t;


class SomeDataObject {
public:
   const Floats_t& GetFloats() { return m_floats; }
   const Tuples_t& GetTuples() { return m_tuples; }

public:
   void AddFloat( float f ) { m_floats.push_back( f ); }
   void AddTuple( const std::vector< float >& t ) { m_tuples.push_back( t ); }

private:
   Floats_t m_floats;
   Tuples_t m_tuples;
};


struct SomeDataStruct {
   Floats_t Floats;
   Char_t   Label[3];
   Int_t    NLabel;
};
