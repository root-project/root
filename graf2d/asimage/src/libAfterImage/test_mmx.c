/*
 * Copyright (c) 2000,2001,2004 Sasha Vasko <sasha at aftercode.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#define LOCAL_DEBUG
#define DO_CLOCKING
#ifdef NO_DEBUG_OUTPUT
#undef DEBUG_RECTS
#undef DEBUG_RECTS2
#endif

#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif


#include <string.h>
#ifdef DO_CLOCKING
#if TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif

#ifdef HAVE_MMX
#include <mmintrin.h>
#include <xmmintrin.h>
#endif

#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif
#include "asvisual.h"
#include "blender.h"
#include "asimage.h"
#include "imencdec.h"

#define TEST_PADDD
#define USE_PREFETCH

#ifdef DO_CLOCKING
#define MIN_TEST_LEN 15000000
#define MAX_TEST_LEN 15000001
#define MAX_REPS 1
#else
#define MIN_TEST_LEN 1
#define MAX_TEST_LEN 10001
#define MAX_REPS 1
#endif

static CARD32 rnd32_seed = 345824357;

#define MAX_MY_RND32		0x00ffffffff
#ifdef WORD64
#define MY_RND32() \
(rnd32_seed = ((1664525L*rnd32_seed)&MAX_MY_RND32)+1013904223L)
#else
#define MY_RND32() \
(rnd32_seed = (1664525L*rnd32_seed)+1013904223L)
#endif


int main()
{
	int test_len ; 
	CARD32 *test_set1 ;
	CARD32 *test_set2 ;	
	CARD32 *control_data ;	
	int i, reps ; 

	for( test_len = MIN_TEST_LEN ; test_len < MAX_TEST_LEN ; ++test_len ) 
	{
		test_set1 = safemalloc( (test_len + (test_len&0x01))* sizeof(CARD32) );	
		test_set2 = safemalloc( (test_len + (test_len&0x01))* sizeof(CARD32) );
		control_data = safemalloc( (test_len + (test_len&0x01))* sizeof(CARD32) );			
		for( i = 0 ; i < test_len ; ++i ) 
		{
			test_set1[i] = MY_RND32()& 0x00FFFFFF ;
			test_set2[i] = MY_RND32()& 0x00FFFFFF ;
		}
		{
			START_TIME(int_math);
			for( reps = 0 ; reps < MAX_REPS ; ++reps ) 
			{
				for( i = 0 ; i < test_len ; ++i ) 
				{
#ifdef TEST_PADDD				
					control_data[i] = test_set1[i] + test_set2[i] ; 
#else
					control_data[i] = test_set2[i] >> 1 ; 
#endif
				}		
			}
			SHOW_TIME("Standard int math : ", int_math);
		}
		{
			START_TIME(mmx_math);
			for( reps = 0 ; reps < MAX_REPS ; ++reps ) 
			{
				int len = test_len + (test_len&0x00000001); 
				__m64  *vdst = (__m64*)&(test_set1[0]);
				__m64  *vinc = (__m64*)&(test_set2[0]);
				__m64  *vsrc = (__m64*)&(test_set2[0]);
				len = len>>1;
				i = 0 ; 
				do{
#ifdef TEST_PADDD				
					vdst[i] = _mm_add_pi32(vdst[i],vinc[i]);  /* paddd */
#ifdef USE_PREFETCH
					_mm_prefetch( &vinc[i+16], _MM_HINT_NTA );
#endif
#else
					vdst[i] = _mm_srli_pi32(vsrc[i],1);  /* psrld */
#endif 
				}while( ++i < len );
			}
			SHOW_TIME("MMX int math : ", mmx_math);
		}
		for( i = 0 ; i < test_len ; ++i ) 
			if( control_data[i] != test_set1[i] ) 
				fprintf( stderr, "test %d: position %d differs - %8.8lX	and %8.8lX, set2 = %8.8lX\n", test_len, i, control_data[i], test_set1[i], test_set2[i] );
//			else	fprintf( stderr, "test %d: position %d same    - %8.8lX	and %8.8lX\n", test_len, i, control_data[i], test_set1[i] );
		
		free( control_data );
		free( test_set2 );
		free( test_set1 );
	}
}
