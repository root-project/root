#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H

// utility functions to fill with random data

template<class V>
void fillRandomVec(TRandom & r, V  & v, unsigned int len, unsigned int start = 0, double offset = 1) {
  for(unsigned int i = start; i < len+start; ++i)
      v[i] = r.Rndm() + offset;
}

template<class M>
void fillRandomMat(TRandom & r, M  & m, unsigned int first, unsigned int second, unsigned int start = 0, double offset = 1) {
  for(unsigned int i = start; i < first+start; ++i)
    for(unsigned int j = start; j < second+start; ++j)
      m(i,j) = r.Rndm() + offset;
}

template<class M>
void fillRandomSym(TRandom & r, M  & m, unsigned int first, unsigned int start = 0, double offset = 1) {
  for(unsigned int i = start; i < first+start; ++i) { 
    for(unsigned int j = i; j < first+start; ++j) { 
      if ( i != j ) { 
	m(i,j) = r.Rndm() + offset;
	m(j,i) = m(i,j);
      }      
      else // add extra offset to make no singular when inverting
	m(i,i) = r.Rndm() + 3*offset;
    }
  }
}

#endif
