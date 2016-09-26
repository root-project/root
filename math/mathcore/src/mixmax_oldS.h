#include <stdio.h>
#include <stdint.h>

#ifndef MIXMAX_H_
#define MIXMAX_H_

// #ifdef __cplusplus
// extern "C" {
// #endif
	
#ifndef _N
#define N 256 
/* The currently recommended N are 3150, 1260, 508, 256, 240, 88
   Since the algorithm is linear in N, the cost per number is almost independent of N.
 */
#else
#define N _N
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
rng_state_t  *rng_copy(myuint *Y);         /* init from vector, takes the vector Y, returns pointer to the newly allocated and initialized state */
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
    myuint mod128(__uint128_t s);
//myuint modmulM61(myuint s, myuint a);
    myuint fmodmulM61(myuint cum, myuint s, myuint a);

#define MERSBASE M61 //xSUFF(M61)
#define MOD_PAYNE(k) ((((k)) & MERSBASE) + (((k)) >> BITS) )  // slightly faster than my old way, ok for addition
#define MOD_REM(k) ((k) % MERSBASE )  // latest Intel CPU is supposed to do this in one CPU cycle, but on my machines it seems to be 20% slower than the best tricks
#define MOD_MERSENNE(k) MOD_PAYNE(k)

#define INV_MERSBASE (0x1p-61)


// the charpoly is irreducible for the combinations of N and SPECIAL and has maximal period for N=508, 256, half period for 1260, and 1/12 period for 3150

#if (N==256)
#define SPECIAL -1
#define MOD_MULSPEC(k) (MERSBASE-(k))

#elif (N==3150)
#define SPECIAL -11
#define MOD_MULSPEC(k) (( ( ( 6*(MERSBASE-(k) )) % MERSBASE) + ( ( 5*(MERSBASE-(k) )) % MERSBASE) ) % MERSBASE )

#elif (N==1000 || N==1051 || N==44)
#define SPECIAL 0
#define MOD_MULSPEC(k) (0)

#elif (N==88 || N==40)
#define SPECIAL 1
#define MOD_MULSPEC(k) (k)

#else
#define SPECIAL -1
#define MOD_MULSPEC(k)  (MERSBASE-(k) )

#endif // list of interesting N for modulus M61 ends here

#ifndef __MIXMAX_C // c++ can put code into header files, why cant we? (with the inline declaration, should be safe from duplicate-symbol error)
	
#define get_next(X) GET_BY_MACRO(X)

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
    
#define get_next_float(X) get_next_float_BY_MACRO(X)

	
inline double get_next_float_BY_MACRO(rng_state_t* X){
        int64_t Z=(int64_t)get_next(X);
#ifdef __SSE__
#warning using SSE inline assemly for int64 -> double conversion, not really necessary in GCC-5 or better
    double F;
        __asm__ ("pxor %0, %0;"
                 "cvtsi2sdq %1, %0;"
                 :"=x"(F)
                 :"r"(Z)
                 );
       return F*INV_MERSBASE;
#else
       return Z*INV_MERSBASE;
#endif
    }

#endif // __MIXMAX_C
	

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

const gsl_rng_type *gsl_rng_ran3 = &mixmax_type;


#endif // HOOKUP_GSL

#endif // closing MIXMAX_H_
