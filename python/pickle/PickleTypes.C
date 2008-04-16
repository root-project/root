/*
  File: roottest/python/pickle/PickleTypes.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 04/16/08
  Last: 04/16/08

  Note: these started off as the same types as ../ttree/TTreeTypes.C
*/

#include <vector>


#ifdef __MAKECINT__
#pragma link C++ class vector< vector< float > >;
#pragma link C++ class vector< vector< float > >::iterator;
#pragma link C++ class vector< vector< float > >::const_iterator;
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
