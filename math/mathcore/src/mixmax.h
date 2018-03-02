/*
 *  mixmax.c
 *  A Pseudo-Random Number Generator
 *
 *  Created by Konstantin Savvidy.
 *
 *  The code is released under GNU Lesser General Public License v3
 *
 *	G.K.Savvidy and N.G.Ter-Arutyunian,
 *  On the Monte Carlo simulation of physical systems,
 *	J.Comput.Phys. 97, 566 (1991);
 *  Preprint EPI-865-16-86, Yerevan, Jan. 1986
 *
 *  K.Savvidy
 *  The MIXMAX random number generator
 *  Comp. Phys. Commun. 196 (2015), pp 161–165
 *  http://dx.doi.org/10.1016/j.cpc.2015.06.003
 *
 */

#ifndef MIXMAX_H_
#define MIXMAX_H_

//#define USE_INLINE_ASM

//#ifdef __cplusplus
//extern "C" {
//#endif

#ifndef ROOT_MM_N
#define N 240
/* The currently recommended generator is the three-parameter MIXMAX with
N=240, s=487013230256099140, m=2^51+1

   Other newly recommended N are N=8, 17 and 240, 
   as well as the ordinary MIXMAX with N=256 and s=487013230256099064

   Since the algorithm is linear in N, the cost per number is almost independent of N.
 */
#else
#define N ROOT_MM_N
#endif

#ifndef __LP64__
typedef uint64_t myuint;
//#warning but no problem,  'myuint' is 'uint64_t'
#else
typedef unsigned long long int myuint;
//#warning but no problem,  'myuint' is 'unsigned long long int'
#endif

struct rng_state_st
{
    myuint V[N];
    myuint sumtot;
    int counter;
    FILE* fh;
};

typedef struct rng_state_st rng_state_t; // C struct alias

int  rng_get_N(void); // get the N programmatically, useful for checking the value for which the library was compiled

rng_state_t  *rng_alloc();                 /* allocate the state */
int           rng_free(rng_state_t* X);    /* free memory occupied by the state */
rng_state_t  *rng_copy(myuint *Y);         /* init from vector, takes the vector Y, 
                                               returns pointer to the newly allocated and initialized state */
void read_state(rng_state_t* X, const char filename[] );
void print_state(rng_state_t* X);
    int iterate(rng_state_t* X);
    myuint iterate_raw_vec(myuint* Y, myuint sumtotOld);


//   FUNCTIONS FOR SEEDING

typedef uint32_t myID_t;

void seed_uniquestream(rng_state_t* X, myID_t clusterID, myID_t machineID, myID_t runID, myID_t  streamID );
/*
 best choice: will make a state vector from which you can get at least 10^100 numbers 
 guaranteed mathematically to be non-colliding with any other stream prepared from another set of 32bit IDs,
 so long as it is different by at least one bit in at least one of the four IDs
			-- useful if you are running a parallel simulation with many clusters, many CPUs each
 */

void seed_spbox(rng_state_t* X, myuint seed);    // non-linear method, makes certified unique vectors,  probability for streams to collide is < 1/10^4600

void seed_vielbein(rng_state_t* X, unsigned int i); // seeds with the i-th unit vector, i = 0..N-1,  for testing only



//   FUNCTIONS FOR GETTING RANDOM NUMBERS

#ifdef __MIXMAX_C
	myuint get_next(rng_state_t* X);         // returns 64-bit int, which is between 1 and 2^61-1 inclusive
	double get_next_float(rng_state_t* X);   // returns double precision floating point number in (0,1]
#endif  //__MIXMAX_C

void fill_array(rng_state_t* X, unsigned int n, double *array); // fastest method: set n to a multiple of N (e.g. n=256)

void iterate_and_fill_array(rng_state_t* X, double *array); // fills the array with N numbers

myuint precalc(rng_state_t* X);
/* needed if the state has been changed by something other than  iterate, but no worries, seeding functions call this for you when necessary */
myuint apply_bigskip(myuint* Vout, myuint* Vin, myID_t clusterID, myID_t machineID, myID_t runID, myID_t  streamID );
// applies a skip of some number of steps calculated from the four IDs
void branch_inplace( rng_state_t* Xin, myID_t* ID ); // almost the same as apply_bigskip, but in-place and from a vector of IDs


#define BITS  61

/* magic with Mersenne Numbers */

#define M61   2305843009213693951ULL

    myuint modadd(myuint foo, myuint bar);
//myuint modmulM61(myuint s, myuint a);
    myuint fmodmulM61(myuint cum, myuint s, myuint a);

#define MERSBASE M61 //xSUFF(M61)
#define MOD_PAYNE(k) ((((k)) & MERSBASE) + (((k)) >> BITS) )  // slightly faster than my old way, ok for addition
#define MOD_REM(k) ((k) % MERSBASE )  // latest Intel CPU is supposed to do this in one CPU cycle, but on my machines it seems to be 20% slower than the best tricks
#define MOD_MERSENNE(k) MOD_PAYNE(k)

//#define INV_MERSBASE (0x1p-61)
#define INV_MERSBASE (0.4336808689942017736029811203479766845703E-18)
//const double INV_MERSBASE=(0.4336808689942017736029811203479766845703E-18); // gives "duplicate symbol" error
    
// the charpoly is irreducible for the combinations of N and SPECIAL

#if (N==256)
#define SPECIALMUL 0
#ifdef USE_MIXMAX_256_NEW
// for 1.1
#define SPECIAL 487013230256099064 // s=487013230256099064, m=1 -- good old MIXMAX
#define MOD_MULSPEC(k) fmodmulM61( 0, SPECIAL , (k) )
#else
// for 1.0
#define SPECIAL -1
#define MOD_MULSPEC(k) (MERSBASE - (k));
#endif
    
#elif (N==8)
#define SPECIALMUL 53 // m=2^53+1
#define SPECIAL 0
    
#elif (N==17)
#define SPECIALMUL 36 // m=2^36+1, other valid possibilities are m=2^13+1, m=2^19+1, m=2^24+1
#define SPECIAL 0
    
#elif (N==40)
#define SPECIALMUL 42 // m=2^42+1
#define SPECIAL 0

#elif (N==60)
#define SPECIALMUL 52 // m=2^52+1
#define SPECIAL 0

#elif (N==96)
#define SPECIALMUL 55 // m=2^55+1
#define SPECIAL 0
    
#elif (N==120)
#define SPECIALMUL 51   // m=2^51+1 and a SPECIAL=+1 (!!!)
#define SPECIAL 1
#define MOD_MULSPEC(k) (k)

#elif (N==240)
#define SPECIALMUL 51   // m=2^51+1 and a SPECIAL=487013230256099140
#define SPECIAL 487013230256099140ULL
#define MOD_MULSPEC(k) fmodmulM61( 0, SPECIAL , (k) )
    
#elif (N==44851)
#define SPECIALMUL 0
#define SPECIAL -3
#define MOD_MULSPEC(k) MOD_MERSENNE(3*(MERSBASE-(k)))


#else
#warning Not a verified N, you are on your own!
#define SPECIALMUL 58
#define SPECIAL 0
    
#endif // list of interesting N for modulus M61 ends here


#ifndef __MIXMAX_C // c++ can put code into header files, why cant we? (with the inline declaration, should be safe from duplicate-symbol error)
	
#define get_next(X) GET_BY_MACRO(X)
#define get_next_float(X) get_next_float_BY_MACRO(X)

#endif // __MIXMAX_C
    

inline 	myuint GET_BY_MACRO(rng_state_t* X) {
    int i;
    i=X->counter;
    
    if (i<=(N-1) ){
        X->counter++;
        return X->V[i];
    }else{
        X->sumtot = iterate_raw_vec(X->V, X->sumtot);
        X->counter=2;
        return X->V[1];
    }
}
	
inline double get_next_float_BY_MACRO(rng_state_t* X){
    /* cast to signed int trick suggested by Andrzej Görlich     */
    int64_t Z=(int64_t)GET_BY_MACRO(X);
    double F;
#if defined(__GNUC__) && (__GNUC__ < 5) && (!defined(__ICC)) && defined(__x86_64__) && defined(__SSE2_MATH__) && defined(USE_INLINE_ASM)
//#warning Using the inline assembler
/* using SSE inline assemly to zero the xmm register, just before int64 -> double conversion,
   not really necessary in GCC-5 or better, but huge penalty on earlier compilers 
 */
   __asm__  __volatile__("pxor %0, %0; "
                        :"=x"(F)
                        );
#endif
    F=Z;
    return F*INV_MERSBASE;
}


// ERROR CODES - exit() is called with these
#define ARRAY_INDEX_OUT_OF_BOUNDS   0xFF01
#define SEED_WAS_ZERO               0xFF02
#define ERROR_READING_STATE_FILE    0xFF03
#define ERROR_READING_STATE_COUNTER       0xFF04
#define ERROR_READING_STATE_CHECKSUM      0xFF05

// #ifdef __cplusplus
// }
// #endif

//#define HOOKUP_GSL 1

#ifdef HOOKUP_GSL // if you need to use mixmax through GSL, pass -DHOOKUP_GSL=1 to the compiler
#ifndef __MIXMAX_C

#include <gsl/gsl_rng.h>
unsigned long gsl_get_next(void *vstate);
double gsl_get_next_float(void *vstate);
void seed_for_gsl(void *vstate, unsigned long seed);

static const gsl_rng_type mixmax_type =
{"MIXMAX",                        /* name */
    MERSBASE,                         /* RAND_MAX */
    1,                                /* RAND_MIN */
    sizeof (rng_state_t),
    &seed_for_gsl,
    &gsl_get_next,
    &gsl_get_next_float
};

unsigned long gsl_get_next(void *vstate) {
    rng_state_t* X= (rng_state_t*)vstate;
    return (unsigned long)get_next(X);
}

double gsl_get_next_float(void *vstate) {
    rng_state_t* X= (rng_state_t*)vstate;
    return ( (double)get_next(X)) * INV_MERSBASE;
}

void seed_for_gsl(void *vstate, unsigned long seed){
    rng_state_t* X= (rng_state_t*)vstate;
    seed_spbox(X,(myuint)seed);
}

const gsl_rng_type *gsl_rng_mixmax = &mixmax_type;


#endif // HOOKUP_GSL
#endif // not inside __MIXMAX_C
#endif // closing MIXMAX_H_
