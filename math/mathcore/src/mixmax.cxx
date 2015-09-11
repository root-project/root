// $Id:$
// -*- C++ -*-
//
// -----------------------------------------------------------------------
//            MixMax Matrix PseudoRandom Number Generator
//                        --- MixMax ---
//                       class header file
// -----------------------------------------------------------------------
//
//
//  Created by Konstantin Savvidy on Sun Feb 22 2004.
//  As of version 0.99 and later, the code is being released under
//  GNU Lesser General Public License v3
//
//	Generator described in 
//	N.Z.Akopov, G.K.Savvidy and N.G.Ter-Arutyunian, Matrix Generator of Pseudorandom Numbers, 
//	J.Comput.Phys. 97, 573 (1991); 
//	Preprint EPI-867(18)-86, Yerevan Jun.1986;
//
//  and
//
//  K.Savvidy
//  The MIXMAX random number generator
//  Comp. Phys. Commun. (2015)
//  http://dx.doi.org/10.1016/j.cpc.2015.06.003
//
// -----------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define __MIXMAX_C  // do NOT define it in your own program, just include mixmax.h

#include "mixmax.h"


int iterate(rng_state_t* X){
	X->sumtot = iterate_raw_vec(X->V, X->sumtot);
	return 0;
}

#if (SPECIALMUL!=0)
inline uint64_t MULWU (uint64_t k){ return (( (k)<<(SPECIALMUL) & M61) | ( (k) >> (BITS-SPECIALMUL))  )  ;}
#elif (SPECIALMUL==0)
inline uint64_t MULWU (uint64_t k){ (void)k; return 0;}
#else
#error SPECIALMUL not undefined
#endif

myuint iterate_raw_vec(myuint* Y, myuint sumtotOld){
	// operates with a raw vector, uses known sum of elements of Y
	int i;
#ifdef SPECIAL
    myuint temp2 = Y[1];
#endif
	myuint  tempP, tempV;
    Y[0] = ( tempV = sumtotOld);
    myuint sumtot = Y[0], ovflow = 0; // will keep a running sum of all new elements (except Y[0])
	tempP = 0;              // will keep a partial sum of all old elements (except Y[0])
	for (i=1; i<N; i++){
#if (SPECIALMUL!=0)
        myuint tempPO = MULWU(tempP);
        tempP = modadd(tempP,Y[i]);
        tempV = MOD_MERSENNE(tempV + tempP + tempPO); // edge cases ?
#else
        tempP = modadd(tempP , Y[i]);
        tempV = modadd(tempV , tempP);
#endif
        Y[i] = tempV;
		sumtot += tempV; if (sumtot < tempV) {ovflow++;}
	}
#ifdef SPECIAL
    temp2 = MOD_MULSPEC(temp2);
    Y[2] = modadd( Y[2] , temp2 );
    sumtot += temp2; if (sumtot < temp2) {ovflow++;}
#endif
	return MOD_MERSENNE(MOD_MERSENNE(sumtot) + (ovflow <<3 ));
}

myuint get_next(rng_state_t* X) {
    int i;
    i=X->counter;
    
    if (i<(N) ){
        X->counter++;
        return X->V[i];
    }else{
        X->sumtot = iterate_raw_vec(X->V, X->sumtot);
        X->counter=1;
        return X->V[0];
    }
}

double get_next_float(rng_state_t* X){
    /* cast to signed int trick suggested by Andrzej Görlich     */
    // times for sknuth_Gap test: N =  1,  n = 500000000,  r =  0,   Alpha =        0,   Beta  =   0.0625
    //return ((int64_t)get_next(X)) * INV_MERSBASE;  // 00:01:17.23
    //return ( (double)get_next(X)) * INV_MERSBASE; // 00:01:57.89
    int64_t Z=(int64_t)get_next(X);
#ifdef __SSE__
    double F;
    __asm__ ("pxor %0, %0; "
             "cvtsi2sdq %1, %0; "
             :"=x"(F)
             :"r"(Z)
             );
    return F*INV_MERSBASE;
#else
    return Z*INV_MERSBASE;
#endif

}

void fill_array(rng_state_t* X, unsigned int n, double *array)
{
    // Return an array of n random numbers uniformly distributed in (0,1]
    unsigned int i,j;
    const int M=N-1;
    for (i=0; i<(n/M); i++){
        iterate_and_fill_array(X, array+i*M);
    }
    unsigned int rem=(n % M);
    if (rem) {
        iterate(X);
        for (j=0; j< (rem); j++){
            array[M*i+j] = (int64_t)X->V[j] * (double)(INV_MERSBASE);
        }
        X->counter = j; // needed to continue with single fetches from the exact spot, but if you only use fill_array to get numbers then it is not necessary
    }else{
        X->counter = N;
    }
}

void iterate_and_fill_array(rng_state_t* X, double *array){
    myuint* Y=X->V;
    int i;
    myuint  tempP, tempV;
#if (SPECIAL != 0)
    myuint temp2 = Y[1];
#endif
    Y[0] = (tempV = modadd(Y[0] , X->sumtot));
    //array[0] = (double)tempV * (double)(INV_MERSBASE);
    myuint sumtot = 0, ovflow = 0; // will keep a running sum of all new elements (except Y[0])
    tempP = 0;             // will keep a partial sum of all old elements (except Y[0])
    for (i=1; i<N; i++){
        tempP = modadd(tempP,Y[i]);
        Y[i] = ( tempV = modadd(tempV,tempP) );
        sumtot += tempV; if (sumtot < tempV) {ovflow++;}
        array[i-1] = (int64_t)tempV * (double)(INV_MERSBASE);
    }
#if (SPECIAL != 0)
    temp2 = MOD_MULSPEC(temp2);
    Y[2] = modadd( Y[2] , temp2 );
    sumtot += temp2; if (sumtot < temp2) {ovflow++;}
#endif
    X->sumtot = MOD_MERSENNE(MOD_MERSENNE(sumtot) + (ovflow <<3 ));
}

myuint modadd(myuint foo, myuint bar){
#if defined(__x86_64__)
    myuint out;
    /* Assembler trick suggested by Andrzej Görlich     */
    __asm__ ("addq %2, %0; "
             "btrq $61, %0; "
             "adcq $0, %0; "
             :"=r"(out)
             :"0"(foo), "r"(bar)
             );
    return out;
#else
    return MOD_MERSENNE(foo+bar);
#endif
}

rng_state_t* rng_alloc()
{
/* allocate the state */
	rng_state_t  *p = (rng_state_t*)malloc(sizeof(rng_state_t)); 
	p->fh=NULL; // by default, set the output file handle to stdout  
	return p;
}

int rng_free(rng_state_t* X) /* free the memory occupied by the state */
{
	free(X);
	return 0;
}

rng_state_t*  rng_copy(myuint *Y)
{
	/* copy the vector stored at Y, and return pointer to the newly allocated and initialized state.  
	 It is the user's responsibility  to make sure that Y is properly allocated with rng_alloc, 
	 then pass Y->V or it can also be an array -- such as myuint Y[N+1] and Y[1]...Y[N] have been set to legal values [0 .. MERSBASE-1]
	 Partial sums on this new state are recalculated, and counter set to zero, so that when get_next is called, 
	 it will output the initial vector before any new numbers are produced, call iterate(X) if you want to advance right away */
	rng_state_t* X = rng_alloc();
    myuint sumtot=0,ovflow=0;
	X->counter = 2;
    int i;
	for ( i=0; i < N; i++){
		X->V[i] = Y[i];
        sumtot += X->V[(i)]; if (sumtot < X->V[(i)]) {ovflow++;}

	}
	X->sumtot = MOD_MERSENNE(MOD_MERSENNE(sumtot) + (ovflow <<3 ));
	return X;
}

void seed_vielbein(rng_state_t* X, unsigned int index)	
{
int i;
	if (index<N){
		for (i=0; i < N; i++){
			X->V[i] = 0;
		}
		X->V[index] = 1;
	}else{
		fprintf(stderr, "Out of bounds index, is not ( 0 <= index < N  )\n"); exit(ARRAY_INDEX_OUT_OF_BOUNDS);
	}
	X->counter = N;  // set the counter to N if iteration should happen right away
	//precalc(X);
    X->sumtot = 1; //(index ? 1:0);
	if (X->fh==NULL){X->fh=stdout;}	
}

void seed_spbox(rng_state_t* X, myuint seed)
{ // a 64-bit LCG from Knuth line 26, in combination with a bit swap is used to seed
	const myuint MULT64=6364136223846793005ULL; 
	int i;
    myuint sumtot=0,ovflow=0;
	if (seed == 0){
		fprintf(stderr, " try seeding with nonzero seed next time!\n");
		exit(SEED_WAS_ZERO);
	}
	
	myuint l = seed;

	//X->V[0] = l & MERSBASE;
	if (X->fh==NULL){X->fh=stdout;} // if the filehandle is not yet set, make it stdout
	for (i=0; i < N; i++){
		l*=MULT64; l = (l << 32) ^ (l>>32);
		X->V[i] = l & MERSBASE;
        sumtot += X->V[(i)]; if (sumtot < X->V[(i)]) {ovflow++;}
	}
	X->counter = N;  // set the counter to N if iteration should happen right away
    X->sumtot = MOD_MERSENNE(MOD_MERSENNE(sumtot) + (ovflow <<3 ));
}

myuint precalc(rng_state_t* X){
	int i;
	myuint temp;
	temp = 0;
	for (i=0; i < N; i++){
		temp = MOD_MERSENNE(temp + X->V[i]);
	}	
	X->sumtot = temp; 
	return temp;
}


int rng_get_N(void){return N;}

//#define MASK32 0xFFFFFFFFULL


//inline myuint modmulM61(myuint a, myuint b){
//	// my best modmul so far
//	__uint128_t temp;
//	temp = (__uint128_t)a*(__uint128_t)b;
//	return mod128(temp);
//}

#if defined(__x86_64__)
inline myuint mod128(__uint128_t s){
    myuint s1;
    s1 = ( (  ((myuint)s)&MERSBASE )    + (  ((myuint)(s>>64)) * 8 )  + ( ((myuint)s) >>BITS) );
    return	MOD_MERSENNE(s1);
}

inline myuint fmodmulM61(myuint cum, myuint a, myuint b){
	__uint128_t temp;
	temp = (__uint128_t)a*(__uint128_t)b + cum;
	return mod128(temp);
}

#else // on all other platforms, including 32-bit linux, PPC and PPC64 and all Windows
#define MASK32 0xFFFFFFFFULL

inline myuint fmodmulM61(myuint cum, myuint s, myuint a)
{
    register myuint o,ph,pl,ah,al;
    o=(s)*a;
    ph = ((s)>>32);
    pl = (s) & MASK32;
    ah = a>>32;
    al = a & MASK32;
    o = (o & M61) + ((ph*ah)<<3) + ((ah*pl+al*ph + ((al*pl)>>32))>>29) ;
    o += cum;
    o = (o & M61) + ((o>>61));
    return o;
}
#endif

void print_state(rng_state_t* X){
    int j;
    fprintf(X->fh, "mixmax state, file version 1.0\n" );
    fprintf(X->fh, "N=%u; V[N]={", rng_get_N() );
	for (j=0; (j< (rng_get_N()-1) ); j++) {
		fprintf(X->fh, "%llu, ", X->V[j] );
	}
	fprintf(X->fh, "%llu", X->V[rng_get_N()-1] );
    fprintf(X->fh, "}; " );
    fprintf(X->fh, "counter=%u; ", X->counter );
    fprintf(X->fh, "sumtot=%llu;\n", X->sumtot );
}

void read_state(rng_state_t* X, const char filename[] ){
    // a function for reading the state from a file, after J. Apostolakis
    FILE* fin;
    if(  ( fin = fopen(filename, "r") ) ){
        char l=0;
        while ( l != '{' ) { // 0x7B = "{"
            l=fgetc(fin); // proceed until hitting opening bracket
        }
        ungetc(' ', fin);
    }else{
        fprintf(stderr, "mixmax -> read_state: error reading file %s\n", filename);
        exit(ERROR_READING_STATE_FILE);
    }
    
    myuint vecVal;
    //printf("mixmax -> read_state: starting to read state from file\n");
    if (!fscanf(fin, "%llu", &X->V[0]) ) {fprintf(stderr, "mixmax -> read_state: error reading file %s\n", filename); exit(ERROR_READING_STATE_FILE);}
     //printf("V[%d] = %llu\n",0, X->V[0]);
    int i;
    for( i = 1; i < rng_get_N(); i++){
        if (!fscanf(fin, ", %llu", &vecVal) ) {fprintf(stderr, "mixmax -> read_state: error reading vector component i=%d from file %s\n", i, filename); exit(ERROR_READING_STATE_FILE);}
        //printf("V[%d] = %llu\n",i, vecVal);
        if(  vecVal <= MERSBASE ){
            X->V[i] = vecVal;
        }else{
            fprintf(stderr, "mixmax -> read_state: Invalid state vector value= %llu"
                    " ( must be less than %llu ) "
                    " obtained from reading file %s\n"
                    , vecVal, MERSBASE, filename);
            
        }
    }
    
    unsigned int counter;
    if (!fscanf( fin, "}; counter=%u; ", &counter)){fprintf(stderr, "mixmax -> read_state: error reading counter from file %s\n", filename); exit(ERROR_READING_STATE_FILE);}
    if( counter <= N ) {
        X->counter= counter;
    }else{
        fprintf(stderr, "mixmax -> read_state: Invalid counter = %d"
                "  Must be 0 <= counter < %u\n" , counter, N);
        print_state(X);
        exit(ERROR_READING_STATE_COUNTER);
    }
    precalc(X);
    myuint sumtot;
    if (!fscanf( fin, "sumtot=%llu\n", &sumtot)){fprintf(stderr, "mixmax -> read_state: error reading checksum from file %s\n", filename); exit(ERROR_READING_STATE_FILE);}
    
    if (X->sumtot != sumtot) {
        fprintf(stderr, "mixmax -> checksum error while reading state from file %s - corrupted?\n", filename);
        exit(ERROR_READING_STATE_CHECKSUM);
    }
//    else{fprintf(stderr, "mixmax -> read_state: checksum ok: %llu == %llu\n",X->sumtot,  sumtot);}
    fclose(fin);
}


#define FUSEDMODMULVEC \
{ for (i =0; i<N; i++){         \
cum[i] =  fmodmulM61( cum[i], coeff ,  Y[i] ) ; \
} }


#define SKIPISON 1

#if ( ( (N==88)||(N==256) ||(N==1000) || (N==3150) ) && BITS==61 && SKIPISON!=0)
void seed_uniquestream( rng_state_t* Xin, myID_t clusterID, myID_t machineID, myID_t runID, myID_t  streamID ){
		seed_vielbein(Xin,0);
		Xin->sumtot = apply_bigskip(Xin->V, Xin->V,  clusterID,  machineID,  runID,   streamID );
	if (Xin->fh==NULL){Xin->fh=stdout;} // if the filehandle is not yet set, make it stdout
}

void branch_inplace( rng_state_t* Xin, myID_t* IDvec ){
	Xin->sumtot = apply_bigskip(Xin->V, Xin->V,  IDvec[3],  IDvec[2],  IDvec[1],   IDvec[0] );
}

myuint apply_bigskip(myuint* Vout, myuint* Vin, myID_t clusterID, myID_t machineID, myID_t runID, myID_t  streamID ){
	/*
	 makes a derived state vector, Vout, from the mother state vector Vin
	 by skipping a large number of steps, determined by the given seeding ID's
	 
	 it is mathematically guaranteed that the substreams derived in this way from the SAME (!!!) Vin will not collide provided
	 1) at least one bit of ID is different
	 2) less than 10^100 numbers are drawn from the stream 
	 (this is good enough : a single CPU will not exceed this in the lifetime of the universe, 10^19 sec, 
	 even if it had a clock cycle of Planch time, 10^44 Hz )
	 
	 Caution: never apply this to a derived vector, just choose some mother vector Vin, for example the unit vector by seed_vielbein(X,0), 
	 and use it in all your runs, just change runID to get completely nonoverlapping streams of random numbers on a different day.
	 
	 clusterID and machineID are provided for the benefit of large organizations who wish to ensure that a simulation 
	 which is running in parallel on a large number of  clusters and machines will have non-colliding source of random numbers.
	 
	 did i repeat it enough times? the non-collision guarantee is absolute, not probabilistic
	 
	 */
	
	
	const	myuint skipMat[128][N] = 
	
#if (N==88) 
#include "mixmax_skip_N88.icc"  // to make this file, delete all except some chosen 128 rows of the coefficients table
#elif (N==256) 
#include "mixmax_skip_N256.icc"
#elif (N==1000) 
#include "mixmax_skip_N1000.icc"
#elif (N==3150) 
#include "mixmax_skip_N3150.icc"
#endif
	;
	
	myID_t IDvec[4] = {streamID, runID, machineID, clusterID};
	int r,i,j,  IDindex;
	myID_t id;
	myuint Y[N], cum[N];
	myuint coeff;
	myuint* rowPtr;
    myuint sumtot=0;
	

    for (i=0; i<N; i++) { Y[i] = Vin[i]; sumtot = modadd( sumtot, Vin[i]); } ;
	for (IDindex=0; IDindex<4; IDindex++) { // go from lower order to higher order ID
		id=IDvec[IDindex];
		//printf("now doing ID at level %d, with ID = %d\n", IDindex, id);     
		r = 0;
		while (id){
			if (id & 1) { 
				rowPtr = (myuint*)skipMat[r + IDindex*8*sizeof(myID_t)];
				//printf("free coeff for row %d is %llu\n", r, rowPtr[0]);
				for (i=0; i<N; i++){ cum[i] = 0; }    
				for (j=0; j<N; j++){              // j is lag, enumerates terms of the poly
					// for zero lag Y is already given
					coeff = rowPtr[j]; // same coeff for all i
					FUSEDMODMULVEC;
					sumtot = iterate_raw_vec(Y, sumtot); 
				}
                sumtot=0;
                for (i=0; i<N; i++){ Y[i] = cum[i]; sumtot = modadd( sumtot, cum[i]); } ;
			}
		id = (id >> 1); r++; // bring up the r-th bit in the ID		
		}		
	}
    sumtot=0;
	for (i=0; i<N; i++){ Vout[i] = Y[i]; sumtot = modadd( sumtot, Y[i]); } ;  // returns sumtot, and copy the vector over to Vout
	return (sumtot) ;
}
#else
#warning For this N, we dont have the skipping coefficients yet, using alternative method to seed

void seed_uniquestream( rng_state_t* Xin, myID_t clusterID, myID_t machineID, myID_t runID, myID_t  streamID ){
    Xin->V[0] = (myuint)clusterID;
    Xin->V[1] = (myuint)machineID;
    Xin->V[2] = (myuint)runID;
    Xin->V[3] = (myuint)streamID;
    Xin->V[4] = (myuint)clusterID << 5;
    Xin->V[5] = (myuint)machineID << 7;
    Xin->V[6] = (myuint)runID     << 11;
    Xin->V[7] = (myuint)streamID  << 13;
    precalc(Xin);
    Xin->sumtot = iterate_raw_vec(Xin->V, Xin->sumtot);
    Xin->sumtot = iterate_raw_vec(Xin->V, Xin->sumtot);
}
#endif // SKIPISON




