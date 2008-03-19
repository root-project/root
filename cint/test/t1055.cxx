/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*  Knuth's best random generator, see also Press et al, Numerical Recipes
    Knuth  The Art of Computer Programming Vol 2, 2nd edition, Sec 5.6
    The Code may be done in floating point arithmetic as well
*/
#include <iostream>
  using std::cout;  using std::endl;

double Random( int Seed = 0 )
{ const int  Rand_Max = 1000000000,  // large even number < INT_MAX
             RSeed    =  161803398;
  const double  Scale = 1.0/Rand_Max;
  
  static int RList[55];  // don't change this !
  static int next, nextp = -1;
  int R, S, i, j, k;
  
  if ( !(nextp+1)  ||  Seed < 0 )  // init the generator (must be called)
  { if ( (R= (RSeed+Seed) % Rand_Max) < 0 ) R+= Rand_Max;
cout << "Rand_Max= " << Rand_Max << endl
     << "Scale= " << Scale << endl;
    RList[54]= R;
    S= 1;
    for ( i= 1; i < 55; i++ )
    { j= (21*i)%55 - 1;
      RList[j]= S;
      if ( (S= R-S) < 0 ) S+= Rand_Max;
      R= RList[j];
    }
    for ( k= 1; k <= 4; k++ )
      for ( i= 0; i < 55; i++ )
        if ( (RList[i]-= RList[(i+31)%55]) < 0 ) RList[i]+= Rand_Max;
        
    next= -1;  nextp= 30;
  } // end of initialization
  
  if ( ++next  == 55 ) next= 0;
  if ( ++nextp == 55 ) nextp= 0;
  
  if ( (R= RList[next] - RList[nextp]) < 0 ) R+= Rand_Max;
  RList[next]= R;
  
  return  R*Scale;
}


int main() {
  double R;
  for (int i=1; i <= 10; ++i ) {
    R= Random();
    cout << R << endl;
  }
  return 0;
}
