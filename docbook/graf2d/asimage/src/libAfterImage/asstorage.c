/* This file contains code for memopry management for image data    */
/********************************************************************/
/* Copyright (c) 2004 Sasha Vasko <sasha at aftercode.net>          */
/********************************************************************/
/*
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#undef LOCAL_DEBUG
/* #undef NO_DEBUG_OUTPUT */
#ifndef NO_DEBUG_OUTPUT
#undef DEBUG_COMPRESS
#undef DEBUG_THRESHOLD
#endif
#define DO_CLOCKING

#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif

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
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <memory.h>

#ifndef HAVE_ZLIB_H
#include "zlib/zlib.h"
#else
#include <zlib.h>
#endif

#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif

#include "asstorage.h"

/* default storage : */

ASStorage *_as_default_storage = NULL ;

#define get_default_asstorage()   (_as_default_storage?_as_default_storage:(_as_default_storage=create_asstorage()))


/************************************************************************/
/* Private Functions : 													*/
/************************************************************************/

#define StorageID2BlockIdx(id)    	(((((CARD32)(id))>>14)&0x0003FFFF)-1)
#define StorageID2SlotIdx(id)    	 ((((CARD32)(id))&0x00003FFF)-1)

static size_t UsedMemory = 0 ;
static size_t UncompressedSize = 0, CompressedSize = 0 ;

static inline ASStorageID 
make_asstorage_id( int block_id, int slot_id )
{
	ASStorageID id = 0 ;
	if( block_id > 0 && block_id < (0x01<<18)&& slot_id > 0 && slot_id < (0x01<<14)) 
		id = ((CARD32)block_id<<14)|(CARD32)slot_id ;
	return id;
}

static int 
rlediff_compress_bitmap8( CARD8 *buffer,  CARD8* data, int size, CARD32 bitmap_threshold )
{	
	int i = 0, comp_size = 0, last_val = 0 ;
	while( i < size ) 
	{
		int count = 0 ;
		do{	
/*			LOCAL_DEBUG_OUT( "data[%d] = %d, threshold = %d\n", i, data[i], bitmap_threshold); */
			if( (( data[i] >= bitmap_threshold )?1:0) != last_val ) 	  
				break; 
			++i ;
		}while( ++count < 255 && i < size ); 	 
		last_val = (last_val == 1)?0:1 ;
		buffer[comp_size++] = count ;
	}
	return comp_size;	
}	 

static int 
rlediff_compress_bitmap32( CARD8 *buffer,  CARD8* data, int size, CARD32 bitmap_threshold )
{	
	int i = 0, comp_size = 0, last_val = 0 ;
	CARD32 *data32 = (CARD32*)data ;
	while( i < size ) 
	{
		int count = 0 ;
		do{	
			if( (( data32[i] >= bitmap_threshold )?1:0) != last_val ) 	  
				break; 
			++i ;
		}while( ++count < 255 && i < size ); 	 
		last_val = (last_val == 1)?0:1 ;
		buffer[comp_size++] = count ;
	}
	return comp_size;	
}	 


static void
compute_diff8( register ASStorageDiff *diff, register CARD8 *data, int size ) 
{
	register int i = 0;	
	diff[0] = data[0] ;
/*	fprintf( stderr, "%d(%4.4X) ", diff[0], diff[0] ); */
	while( ++i < size ) 
	{	
		diff[i] = (ASStorageDiff)data[i] - (ASStorageDiff)data[i-1] ;
/*		fprintf( stderr, "%d(%4.4X) ", diff[i], diff[i] ); */
	}
/*	fprintf( stderr, "\n" ); */
}	   

static void
compute_diff32( register ASStorageDiff *diff, CARD8 *data, int size ) 
{
	register int i = 0;	
	register CARD32 *data32 = (CARD32*)data ;
	diff[0] = data32[0] ;
/*	fprintf( stderr, "\n0:%d(%4.4X) ", diff[0], diff[0] ); */
	while( ++i < size ) 
	{	
		diff[i] = (ASStorageDiff)data32[i] - (ASStorageDiff)data32[i-1] ;
/*		fprintf( stderr, "%d:%d(%4.4X) ", i, diff[i], diff[i] ); */
	}
/*	fprintf( stderr, "\n" ); */
}	   

static void
compute_diff32_8bitshift( register ASStorageDiff *diff, CARD8 *data, int size ) 
{
	register int i = 0;	
	register CARD32 *data32 = (CARD32*)data ;
	register ASStorageDiff dp = data32[0]>>8;
	diff[0] = dp ;
	while( ++i < size ) 
	{
		register ASStorageDiff d = data32[i]>>8;
		diff[i] = d - dp ;
		dp = d;
	}
}	   

static void
compute_diff32_16bitshift( register ASStorageDiff *diff, CARD8 *data, int size ) 
{
	register int i = 0;	
	register CARD32 *data32 = (CARD32*)data ;
	register ASStorageDiff dp = data32[0]>>16;
	diff[0] = dp ;
	while( ++i < size ) 
	{
		register ASStorageDiff d = data32[i]>>16;
		diff[i] = d - dp ;
		dp = d;
	}
}	   

static void
compute_diff32_masked( register ASStorageDiff *diff, CARD8 *data, int size ) 
{
	register int i = 0;	
	register CARD32 *data32 = (CARD32*)data ;
	register ASStorageDiff dp = data32[0]&0x0ff;
	diff[0] = dp ;
	while( ++i < size ) 
	{
		register ASStorageDiff d = data32[i]&0x0ff;
		diff[i] = d - dp ;
		dp = d;
	}
}	   

static void
compute_diff32_8bitshift_masked( register ASStorageDiff *diff, CARD8 *data, int size ) 
{
	register int i = 0;	
	register CARD32 *data32 = (CARD32*)data ;
	register ASStorageDiff dp = (data32[0]>>8)&0x0ff;
	diff[0] = dp ;
	while( ++i < size ) 
	{
		register ASStorageDiff d = (data32[i]>>8)&0x0ff;
		diff[i] = d - dp ;
		dp = d;
	}
}	   

static void
compute_diff32_16bitshift_masked( register ASStorageDiff *diff, CARD8 *data, int size ) 
{
	register int i = 0;	
	register CARD32 *data32 = (CARD32*)data ;
	register ASStorageDiff dp = (data32[0]>>16)&0x0ff;
	diff[0] = dp ;
	while( ++i < size ) 
	{
		register ASStorageDiff d = (data32[i]>>16)&0x0ff;
		diff[i] = d - dp ;
		dp = d;
	}
}	   

static void
compute_diff32_24bitshift_masked( register ASStorageDiff *diff, CARD8 *data, int size ) 
{
	register int i = 0;	
	register CARD32 *data32 = (CARD32*)data ;
	register ASStorageDiff dp = (data32[0]>>24)&0x0ff;
	diff[0] = dp ;
	while( ++i < size ) 
	{
		register ASStorageDiff d = (data32[i]>>24)&0x0ff;
		diff[i] = d - dp ;
		dp = d;
	}
}	   

static int 
rlediff_compress( CARD8 *buffer,  ASStorageDiff *diff, int size )
{
	int comp_size = 1 ;
	int i = 1;
	
	buffer[0] = (CARD8)diff[0] ; 
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
 	fprintf(stderr, "first byte: 0x%2.2X \n", buffer[0] );
#endif				
	while( i < size ) 
	{
		int run_step = 0;
		int run_size2 = 0;
		int d = diff[i] ; 
		
		if( d == 0 ) 
		{
			int zero_size = 0 ;  /* intentionally ! */ 
			while( ++i < size && zero_size < 127 ) 	 
			{	
				if( diff[i] != 0 ) 
					break;
				++zero_size ;
			}
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
			fprintf( stderr, "comp_size = %d at line %d\n", comp_size, __LINE__ );
#endif
			if( comp_size + 1 > size )
				return 0; 

			buffer[comp_size] = RLE_ZERO_SIG | zero_size ;
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
			fprintf(stderr, "in %d out %d: 0x%2.2X  - %d zeros\n", i, comp_size, buffer[comp_size], zero_size+1 );
#endif
			++comp_size ;
		}else 
		{	
			if( d < 0) d = -d ;
			
			if( d <= 8 )  
			{ /* see if we can pack everything into 2 or 4 bit string */
				do
				{
					if( (d = diff[i]) == 0 ) 
						break;
					if( d < 0) d = -d ;
					if( d > 8 ) 
						break;
					if( run_size2 == run_step ) 
					{	
						if( d > 2 )
						{
							if( run_size2 >= 4 )
								break;
						}else if( ++run_size2  >= 16 ) 
						{
							++i ;	  
							break; 
						}
					}
					++run_step ;
					++i ;
				}while( i < size && run_step < 64 ); 	
				
				if( run_step > run_size2 ) 
				{                  /* encoding as 4 bit values */
					int k = i - run_step;
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
					fprintf( stderr, "comp_size = %d, run_step = %d at line %d\n", comp_size, run_step, __LINE__ );
#endif
					if( comp_size + 1 + run_step/2 > size )
						return 0; 
 

					buffer[comp_size] = RLE_NOZERO_SHORT_SIG | (run_step-1) ;											   
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
					fprintf(stderr, "in %d out %d: 0x%2.2X  - %d 4bits things\n", i, comp_size, buffer[comp_size], run_step );
#endif				
					++comp_size;
					do
					{
						if( (d = diff[k]) < 0 ) buffer[comp_size] = 0x80|((-d-1)<<4) ;
						else					buffer[comp_size] =      ((d-1)<<4) ;
						
						if( ++k < i )
						{	
							if( (d = diff[k]) < 0 ) buffer[comp_size] |= 0x08|(-d-1) ;
							else					buffer[comp_size] |=      (d-1) ;
							++k ;
						}
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
						fprintf(stderr, "0x%2.2X ", buffer[comp_size] );
#endif
						++comp_size ;
					}while( k < i );
				}else	 
				{					/* encoding as 2 bit values */
					int k = i - run_size2;
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
					fprintf( stderr, "comp_size = %d, run_step = %d at line %d\n", comp_size, run_size2, __LINE__ );
#endif
					if( comp_size + 1 + run_size2/4 > size )
						return 0; 
 

					buffer[comp_size] = RLE_NOZERO_LONG1_SIG | (run_size2-1) ;											   
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
					fprintf(stderr, "in %d out %d: 0x%2.2X  - %d 2bits things\n", i, comp_size, buffer[comp_size], run_size2 );
#endif				
					++comp_size;
					do
					{
						if( (d = diff[k]) < 0 ) buffer[comp_size] = 0x80|((-d-1)<<6) ;
						else					buffer[comp_size] =      ((d-1)<<6) ;

						if( ++k < i )
						{	
							if( (d = diff[k]) < 0 ) buffer[comp_size] |= 0x20|((-d-1)<<4) ;
							else					buffer[comp_size] |=      ((d-1)<<4) ;
							if( ++k < i )
							{
								if( (d = diff[k]) < 0 ) buffer[comp_size] |= 0x08|((-d-1)<<2) ;
								else					buffer[comp_size] |=      ((d-1)<<2) ;
								if( ++k < i )
								{	
 									if( (d = diff[k]) < 0 ) buffer[comp_size] |= 0x02|(-d-1) ;
									else					buffer[comp_size] |=      (d-1) ;
									++k ;
								}
							}
						}
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
						fprintf(stderr, "0x%2.2X ", buffer[comp_size] );
#endif
						++comp_size ;
					}while( k  < i );
				}	 
			}else if( d <= 135 )  
			{                      /* 8 bit strings */
				int k = 0;
				do
				{
					if( (d = diff[i]) == 0 ) 
						break;
					if( d < 0) d = -d ;
					if( d > 135 || d <= 8 ) 
						break;
					++run_step ;
				}while( ++i < size && run_step < 16 ); 	
			
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
				fprintf( stderr, "comp_size = %d, run_step = %d, size = %d at line %d\n", comp_size, run_step, size, __LINE__ );
#endif
				if( comp_size + 1 + run_step > size )
					return 0; 


				buffer[comp_size] = RLE_NOZERO_LONG2_SIG | (run_step-1) ;											   
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
				fprintf(stderr, "in %d out %d: 0x%2.2X  - %d 8bits things\n", i, comp_size, buffer[comp_size], run_step );
#endif				
				++comp_size;
				k = i - run_step ;
				do
				{
					if( (d = diff[k]) < 0 ) buffer[comp_size] = 0x80|(-d-8) ;
					else					buffer[comp_size] =      (d-8) ;
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
					fprintf(stderr, "0x%2.2X ", buffer[comp_size] );
#endif
					++comp_size ;
				}while( ++k < i );
			}else		 
			{	
				int k = 0;		/* 9 bit strings */
				do
				{
					if( (d = diff[i]) == 0 ) 
						break;
					if( d < 0) d = -d ;
					if( d <= 135 ) 
						break;
					++run_step ;
				}while( ++i < size && run_step < 16 ); 	

#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
					fprintf( stderr, "comp_size = %d, run_step = %d at line %d\n", comp_size, run_step, __LINE__ );
#endif
				if( comp_size + 1 + run_step > size )
					return 0; 
 
				
				k = i - run_step ;
				if( diff[k] > 0 ) 
					buffer[comp_size] = RLE_9BIT_SIG | (run_step-1) ;											   
				else
					buffer[comp_size] = RLE_9BIT_NEG_SIG | (run_step-1) ;											   
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
				fprintf(stderr, "in %d out %d: 0x%2.2X  - %d 9bit things\n", i, comp_size, buffer[comp_size], run_step );
#endif				
				++comp_size;
				do
				{
					if( (d = diff[k]) < 0 ) buffer[comp_size] = -d ;
					else					buffer[comp_size] =  d ;
					
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
					fprintf(stderr, "0x%2.2X ", buffer[comp_size] );
#endif
					++comp_size ;
				}while( ++k  < i );
			} 
		}
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
	   	fprintf(stderr, "\n");
#endif
	}	 
	/* fprintf( stderr, "compressed from %d to %d\n", size, comp_size ); */
	/* its better to do it here from performance point of view since most of
	 * the data will be well compressed */
	if( comp_size > size ) 
		return 0;
	return comp_size ;
}	 

static int
rlediff_decompress_bitmap( CARD8 *buffer,  CARD8* data, int size, CARD8 bitmap_value )
{
	unsigned int count ;
	int out_bytes = 0 ;
	int in_bytes = 0 ;
	CARD8 curr_val = 0;

	while( in_bytes < size ) 
	{
		count = ((unsigned int)(data[in_bytes++])+1) ;
		while( --count > 0 ) 
			buffer[out_bytes++] = curr_val ;
		curr_val = (curr_val == bitmap_value)? 0 : bitmap_value ;
	}
	
	LOCAL_DEBUG_OUT( "in_bytes = %d, out_bytes = %d, size = %d", in_bytes, out_bytes, size );
	return out_bytes;
}

static int
rlediff_decompress( CARD8 *buffer,  CARD8* data, int size )
{
	int count ;
	int out_bytes = 1 ;
	int in_bytes = 1 ;
	CARD8 last_val;

	buffer[0] = last_val = data[0] ; 

	while( in_bytes < size ) 
	{
		CARD8 c = data[in_bytes++] ;
#if defined(DEBUG_COMPRESS) && !defined(NO_DEBUG_OUTPUT)
		fprintf(stderr, "in %d out %d: 0x%2.2X \n", in_bytes, out_bytes, c);
#endif				

		if( (c & RLE_ZERO_MASK) == 0 ) 			   
		{
			count = (int)c  + 1 ;
			while( --count >= 0 )  	  
				buffer[out_bytes++] = last_val ;
		}else if( (c & RLE_NOZERO_SHORT_MASK ) == RLE_NOZERO_SHORT_SIG ) 
		{
			count = c & RLE_NOZERO_SHORT_LENGTH ;
			++count ;
			while( --count >= 0 ) 
			{
				CARD8 mod = ((data[in_bytes]>>4)&0x07)+1;
				last_val = (data[in_bytes]&0x80)?last_val - mod : last_val + mod ;
				buffer[out_bytes++] = last_val ;
				if( --count >= 0 )
				{
					mod = (data[in_bytes]&0x07)+1;
					last_val = (data[in_bytes]&0x08)?last_val - mod : last_val + mod ;
					buffer[out_bytes++] = last_val ;
				}
				++in_bytes ;
			}
		}else
		{
			count = c & RLE_NOZERO_LONG_LENGTH ;
			++count ;
			if( (c & RLE_NOZERO_LONG_MASK ) == RLE_NOZERO_LONG1_SIG ) 
			{
				while( --count >= 0 ) 
				{
					CARD8 mod = ((data[in_bytes]>>6)&0x01)+1;
					last_val = (data[in_bytes]&0x80)?last_val - mod : last_val + mod ;
					buffer[out_bytes++] = last_val ;
					if( --count >= 0 )
					{
						mod = ((data[in_bytes]>>4)&0x01)+1;
						last_val = (data[in_bytes]&0x20)?last_val - mod : last_val + mod ;
						buffer[out_bytes++] = last_val ;
						if( --count >= 0 )
						{
							mod = ((data[in_bytes]>>2)&0x01)+1;
							last_val = (data[in_bytes]&0x08)?last_val - mod : last_val + mod ;
							buffer[out_bytes++] = last_val ;
							if( --count >= 0 )
							{
								mod = (data[in_bytes]&0x01)+1;
								last_val = (data[in_bytes]&0x02)?last_val - mod : last_val + mod ;
								buffer[out_bytes++] = last_val ;
							}
						}

					}
					++in_bytes ;
				}
			}else if( (c & RLE_NOZERO_LONG_MASK ) == RLE_NOZERO_LONG2_SIG ) 
			{
				while( --count >= 0 ) 
				{
					CARD8 mod = (data[in_bytes]&0x7F)+8;
					last_val = (data[in_bytes]&0x80)?last_val - mod : last_val + mod ;
					buffer[out_bytes++] = last_val ;
					++in_bytes ;
				}
			}else
			{
				Bool sign = ((c & RLE_NOZERO_LONG_MASK ) == RLE_9BIT_NEG_SIG);
				while( --count >= 0 ) 
				{
					CARD8 mod = data[in_bytes];
					last_val = sign? last_val - mod : last_val + mod ;
					sign = !sign ;
					buffer[out_bytes++] = last_val ;
					++in_bytes ;
				}
			}
		}	 
	}	 
	LOCAL_DEBUG_OUT( "in_bytes = %d, out_bytes = %d, size = %d", in_bytes, out_bytes, size );
	return out_bytes;
}	 


static int
copy_data_tinted (CARD8 *buffer, CARD32 *data32, int size, CARD32 tint)
{
	int comp_size = 0;
	do{	buffer[comp_size] = (data32[comp_size]*tint)>>8;}while (++comp_size < size);
	return comp_size;
}

static int
copy_data_tinted_8bitshift (CARD8 *buffer, CARD32 *data32, int size, CARD32 tint)
{
	int comp_size = 0;
	do{	buffer[comp_size] = (data32[comp_size]*tint)>>16;}while (++comp_size < size);
	return comp_size;
}

static int
copy_data_tinted_16bitshift (CARD8 *buffer, CARD32 *data32, int size, CARD32 tint)
{
	int comp_size = 0;
	do{	buffer[comp_size] = (data32[comp_size]*tint)>>24;}while (++comp_size < size);
	return comp_size;
}

static int
copy_data_tinted_masked ( CARD8 *buffer, CARD32 *data32, int size, CARD32 tint)
{
	int comp_size = 0;
	do{	buffer[comp_size] = ((data32[comp_size]&0x0FF)*tint)>>8;}while (++comp_size < size);
	return comp_size;
}

static int
copy_data_tinted_8bitshift_masked (CARD8 *buffer, CARD32 *data32, int size, CARD32 tint)
{
	int comp_size = 0;
	do{	buffer[comp_size] = (((data32[comp_size]>>8)&0x0FF)*tint)>>8;}while (++comp_size < size);
	return comp_size;
}


static int
copy_data_tinted_16bitshift_masked (CARD8 *buffer, CARD32 *data32, int size, CARD32 tint)
{
	int comp_size = 0;
	do{	buffer[comp_size] = (((data32[comp_size]>>16)&0x0FF)*tint)>>8;}while (++comp_size < size);
	return comp_size;
}

static int
copy_data_tinted_24bitshift_masked( CARD8 *buffer, CARD32 *data32, int size, CARD32 tint)
{
	int comp_size = 0;
	do{	buffer[comp_size] = (((data32[comp_size]>>24)&0x0FF)*tint)>>8;}while (++comp_size < size);
	return comp_size;
}

static int
copy_data32 (CARD8 *buffer, CARD32 *data32, int size)
{
	int comp_size = 0;
	do{	buffer[comp_size] = data32[comp_size];}while (++comp_size < size);
	return comp_size;
}

static int
copy_data32_8bitshift (CARD8 *buffer, CARD32 *data32, int size)
{
	int comp_size = 0;
	do{	buffer[comp_size] = data32[comp_size]>>8;}while (++comp_size < size);
	return comp_size;
}

static int
copy_data32_16bitshift (CARD8 *buffer, CARD32 *data32, int size)
{
	int comp_size = 0;
	do{	buffer[comp_size] = data32[comp_size]>>16;}while (++comp_size < size);
	return comp_size;
}

static int
copy_data32_masked ( CARD8 *buffer, CARD32 *data32, int size)
{
	int comp_size = 0;
	do{	buffer[comp_size] = data32[comp_size]&0x0FF;}while (++comp_size < size);
	return comp_size;
}

static int
copy_data32_8bitshift_masked (CARD8 *buffer, CARD32 *data32, int size)
{
	int comp_size = 0;
	do{	buffer[comp_size] = (data32[comp_size]>>8)&0x0FF;}while (++comp_size < size);
	return comp_size;
}


static int
copy_data32_16bitshift_masked (CARD8 *buffer, CARD32 *data32, int size)
{
	int comp_size = 0;
	do{	buffer[comp_size] = (data32[comp_size]>>16)&0x0FF;}while (++comp_size < size);
	return comp_size;
}

static int
copy_data32_24bitshift_masked( CARD8 *buffer, CARD32 *data32, int size)
{
	int comp_size = 0;
	do{	buffer[comp_size] = (data32[comp_size]>>24)&0x0FF;}while (++comp_size < size);
	return comp_size;
}


static CARD8* 
compress_stored_data( ASStorage *storage, CARD8 *data, int size, ASFlagType *flags, int *compressed_size,
					  CARD32 bitmap_threshold )
{
	/* TODO: just a stub for now - need to implement compression */
	int comp_size = size ;
	CARD8  *buffer = data ;
	size_t 	buf_size = size ; 

	static compute_diff_func_type compute_diff_func[2][4] = 
	{	{
			compute_diff32,
			compute_diff32_8bitshift,
			compute_diff32_16bitshift,
			compute_diff32_24bitshift_masked /* to clear the sign bit ! */
		},
		{
			compute_diff32_masked,
			compute_diff32_8bitshift_masked,
			compute_diff32_16bitshift_masked,
			compute_diff32_24bitshift_masked
		}
	};

	static copy_data32_tinted_func_type copy_data32_tinted_func[2][4] = 
	{	{
			copy_data_tinted,
			copy_data_tinted_8bitshift,
			copy_data_tinted_16bitshift,
			copy_data_tinted_24bitshift_masked /* to clear the sign bit ! */
		},
		{
			copy_data_tinted_masked,
			copy_data_tinted_8bitshift_masked,
			copy_data_tinted_16bitshift_masked,
			copy_data_tinted_24bitshift_masked
		}
	};

	static copy_data32_func_type copy_data32_func[2][4] = 
	{	{
			copy_data32,
			copy_data32_8bitshift,
			copy_data32_16bitshift,
			copy_data32_24bitshift_masked /* to clear the sign bit ! */
		},
		{
			copy_data32_masked,
			copy_data32_8bitshift_masked,
			copy_data32_16bitshift_masked,
			copy_data32_24bitshift_masked
		}
	};

	if( size < ASStorageSlot_SIZE ) 
		clear_flags( *flags, ASStorage_RLEDiffCompress );

	if (get_flags( *flags, ASStorage_Bitmap )) /* always compress bitmaps !!!! */
		set_flags( *flags, ASStorage_RLEDiffCompress );

	if( get_flags( *flags, ASStorage_RLEDiffCompress ) )
	{
		int uncompressed_size = size ;

		clear_flags( *flags, ASStorage_RLEDiffCompress );
		if( (int)storage->comp_buf_size < size ) 
		{	
			storage->comp_buf_size = ((size/AS_STORAGE_PAGE_SIZE)+1)*AS_STORAGE_PAGE_SIZE ;
			storage->comp_buf = realloc( storage->comp_buf, storage->comp_buf_size );
			storage->diff_buf = realloc( storage->diff_buf, storage->comp_buf_size*sizeof(ASStorageDiff) );
#ifdef DEBUG_ALLOCS
			show_debug( __FILE__,"compress_stored_data",__LINE__," realloced compression buffer to %d+%d*%d",storage->comp_buf_size, storage->comp_buf_size, sizeof(ASStorageDiff) );
#endif 
		}
		buffer = storage->comp_buf ;
		buf_size = storage->comp_buf_size ;
		if( buffer ) 
		{
			if( get_flags( *flags, ASStorage_Bitmap ) )
			{	
				if( get_flags( *flags, ASStorage_32Bit ) ) 
				{
					uncompressed_size = size / 4 ;
					if( get_flags( *flags, ASStorage_BitShift ) )
						bitmap_threshold = bitmap_threshold<<ASStorage_Flags2Shift(*flags) ;
					comp_size = rlediff_compress_bitmap32( buffer, data, uncompressed_size, bitmap_threshold );
				}else
					comp_size = rlediff_compress_bitmap8( buffer, data, uncompressed_size, bitmap_threshold );
			}else 
			{
				ASStorageDiff tint = bitmap_threshold ;
				if( get_flags( *flags, ASStorage_32Bit ) ) 
				{	
					uncompressed_size = size / 4 ;
					compute_diff_func[get_flags(*flags,ASStorage_Masked)?1:0]
					                 [ASStorage_Flags2ShiftIdx(*flags)](storage->diff_buf, data, uncompressed_size );
				}else
					compute_diff8( storage->diff_buf, data, uncompressed_size ); 	  
				
				if( tint != 255 )
				{
					int i;
					ASStorageDiff *diff = storage->diff_buf ; 
					for( i = 0 ; i < uncompressed_size ; ++i ) 
						diff[i] = (diff[i]*tint)/256 ;
				}	 
				comp_size = rlediff_compress( buffer, storage->diff_buf, uncompressed_size );
			}

			if( comp_size == 0 )	 
			{	
				buffer = data ;
				comp_size = size ;
			}else
			{	
				set_flags( *flags, ASStorage_RLEDiffCompress );
				UncompressedSize += size ;
				CompressedSize += comp_size ;
			}
		}else
			buffer = data ;	 
		
		LOCAL_DEBUG_OUT( "size = %d, compressed_size = %d, flags = 0x%lX", size, comp_size, *flags );
	}	 

	if( buffer == data )
	{
		CARD32 tint = get_flags( *flags, ASStorage_Bitmap )? 0x00FF : bitmap_threshold ;
		if( get_flags( *flags, ASStorage_32Bit ) )
		{
			CARD32 *data32 = (CARD32*)data ;
			size /= 4;
			if( (int)(storage->comp_buf_size) < size ) 
			{	
				storage->comp_buf_size = ((size/AS_STORAGE_PAGE_SIZE)+1)*AS_STORAGE_PAGE_SIZE ;
				storage->comp_buf = realloc( storage->comp_buf, storage->comp_buf_size );
				storage->diff_buf = realloc( storage->diff_buf, storage->comp_buf_size*sizeof(ASStorageDiff) );
#ifdef DEBUG_ALLOCS
			show_debug( __FILE__,"compress_stored_data",__LINE__," realloced compression buffer to %d+%d*%d",storage->comp_buf_size, storage->comp_buf_size, sizeof(ASStorageDiff) );
#endif 

			}
			buffer = storage->comp_buf ;
			if( tint != 0x000000FF ) 
			{	
				copy_data32_tinted_func [get_flags(*flags,ASStorage_Masked)?1:0]
					                 	[ASStorage_Flags2ShiftIdx(*flags)](buffer, data32, size, tint);
			}else
			{
				copy_data32_func [get_flags(*flags,ASStorage_Masked)?1:0]
					           	 [ASStorage_Flags2ShiftIdx(*flags)](buffer, data32, size);

			}	 
		}else if( tint != 0x000000FF ) 
		{
			if( (int)storage->comp_buf_size < size ) 
			{	
				storage->comp_buf_size = ((size/AS_STORAGE_PAGE_SIZE)+1)*AS_STORAGE_PAGE_SIZE ;
				storage->comp_buf = realloc( storage->comp_buf, storage->comp_buf_size );
				storage->diff_buf = realloc( storage->diff_buf, storage->comp_buf_size*sizeof(ASStorageDiff) );
#ifdef DEBUG_ALLOCS
			show_debug( __FILE__,"compress_stored_data",__LINE__," realloced compression buffer to %d+%d*%d",storage->comp_buf_size, storage->comp_buf_size, sizeof(ASStorageDiff) );
#endif 
			}
			buffer = storage->comp_buf ;
			for( comp_size = 0 ; comp_size < size ; ++comp_size )
				buffer[comp_size] = (((CARD32)data[comp_size])*tint)>>8 ;
		}	 
	}	
	if( compressed_size ) 
		*compressed_size = comp_size ;
	return buffer;
}

static CARD8 *
decompress_stored_data( ASStorage *storage, CARD8 *data, int size, int uncompressed_size, 
						ASFlagType flags, CARD8 bitmap_value )
{
	CARD8  *buffer = data ;

	LOCAL_DEBUG_OUT( "size = %d, uncompressed_size = %d, flags = 0x%lX", size, uncompressed_size, flags );
	if( get_flags( flags, ASStorage_RLEDiffCompress ))
	{
		buffer = storage->comp_buf ;
		if( get_flags( flags, ASStorage_Bitmap ) )
			rlediff_decompress_bitmap( buffer, data, size, bitmap_value );	 
		else			
			rlediff_decompress( buffer, data, size );	 
		/* need to check decompressed size */
	}
	
	return buffer;
}

static void
add_storage_slots( ASStorageBlock *block )
{
	int i = block->slots_count ;
	int size ;
	int count = AS_STORAGE_SLOTS_BATCH ;  
	LOCAL_DEBUG_OUT( "block = %p, block->slots = %p", block, block->slots );
	if( block->slots_count + count  >= AS_STORAGE_MAX_SLOTS_CNT ) 
	{
		count = (int)AS_STORAGE_MAX_SLOTS_CNT - block->slots_count ; 
		if( count < 0 ) 
			return;
	}
	block->slots_count += count ; 
	size = block->slots_count*sizeof(ASStorageSlot*)  ;
	LOCAL_DEBUG_OUT( "block->slots_count = %d", block->slots_count );
#ifndef DEBUG_ALLOCS
	LOCAL_DEBUG_OUT( "reallocing %d slots pointers (%d)", block->slots_count, size );
	block->slots = realloc( block->slots, size);
	LOCAL_DEBUG_OUT( "reallocated %d slots pointers", block->slots_count );
#else
	if( block->slots == NULL ) 
		show_debug( __FILE__,"add_storage_slots",__LINE__,"allocating %d slots pointers", block->slots_count );
	else
		show_debug( __FILE__,"add_storage_slots",__LINE__,"reallocating %d slots pointers", block->slots_count );
	block->slots = guarded_realloc( block->slots, block->slots_count*sizeof(ASStorageSlot*));
#endif
	UsedMemory += count*sizeof(ASStorageSlot*) ;
	memset( &(block->slots[i]),	0x00, count*sizeof(ASStorageSlot*) );
}



static ASStorageBlock *
create_asstorage_block( int useable_size )
{
	int allocate_size = (sizeof(ASStorageBlock)+ ASStorageSlot_SIZE + useable_size) ; 
	void *ptr ;	
	ASStorageBlock *block ;

	if( allocate_size%AS_STORAGE_PAGE_SIZE > 0 ) 
		allocate_size = ((allocate_size/AS_STORAGE_PAGE_SIZE)+1)*AS_STORAGE_PAGE_SIZE ;
#ifndef DEBUG_ALLOCS
	ptr = calloc(1,allocate_size);
#else
	{
		char msg[256];
		ptr = guarded_calloc(1,allocate_size);
		sprintf( msg, "allocated %d bytes, block = %p, total used = %d", allocate_size, ptr, UsedMemory + allocate_size );
		PRINT_MEM_STATS(msg);
	}
#endif
	UsedMemory += allocate_size ;
	if( ptr == NULL ) 
		return NULL;
	block = ptr ;
	block->size = allocate_size - sizeof(ASStorageBlock) ;
	block->total_free = block->size - ASStorageSlot_SIZE ;

	block->slots_count = 0 ;
	add_storage_slots( block ) ;   
	
	if( block->slots == NULL ) 
	{	
		free( ptr ); 
		UsedMemory -= allocate_size ;
#ifdef DEBUG_ALLOCS
		show_debug( __FILE__,"create_asstorage_block",__LINE__,"freeing block %p, size = %d, total used = %d", ptr, allocate_size, UsedMemory );
#endif
		return NULL;
	}
	block->start = (ASStorageSlot*)((unsigned char*)ptr+((sizeof(ASStorageBlock)/ASStorageSlot_SIZE)+1)*ASStorageSlot_SIZE);
	block->end = (ASStorageSlot*)((unsigned char*)ptr+(allocate_size-ASStorageSlot_SIZE));
	block->slots[0] = block->start ;
	block->slots[0]->flags = 0 ;  /* slot of the free memory */ 
	block->slots[0]->ref_count = 0 ;
	block->slots[0]->size = ((CARD8*)(block->end) - (CARD8*)(block->start))-ASStorageSlot_SIZE ;
	block->slots[0]->uncompressed_size = block->slots[0]->size ;
	block->slots[0]->index = 0 ;
	block->last_used = 0;
	block->first_free = 0 ;
	
	LOCAL_DEBUG_OUT("Storage block created : block ptr = %p, slots ptr = %p", block, block->slots );
	
	return block;
}

static void
destroy_asstorage_block( ASStorageBlock *block )
{
	UsedMemory -= block->slots_count * sizeof(ASStorageSlot*) ;
	UsedMemory -= block->size + sizeof(ASStorageBlock) ;

#ifndef DEBUG_ALLOCS
	free( block->slots );
	free( block );	  
#else	
	{
		char msg[256];
		sprintf( msg, "freeing block %p, size = %d, total used = %d", block, block->size, UsedMemory );
		guarded_free( block->slots );
		guarded_free( block );
		PRINT_MEM_STATS(msg);
	}
#endif

}

static int
select_storage_block( ASStorage *storage, int compressed_size, ASFlagType flags, int block_id_start )
{
	int i ;
	int new_block = -1 ; 
	compressed_size += ASStorageSlot_SIZE;
	i = block_id_start - 1 ;
	if( i < 0 ) 
		i = 0 ;
	for( ; i < storage->blocks_count ; ++i ) 
	{
		ASStorageBlock *block = storage->blocks[i];
		if( block )
		{	
			if( block->total_free > compressed_size && 
				block->total_free > AS_STORAGE_NOUSE_THRESHOLD && 
				block->last_used+2 < AS_STORAGE_MAX_SLOTS_CNT  )
				return i+1;
		}else if( new_block < 0 ) 
			new_block = i ;
	}		
	/* no available blocks found - need to allocate a new block */
	if( new_block  < 0 ) 
	{
		i = new_block = storage->blocks_count ;
		storage->blocks_count += 16 ;
#ifndef DEBUG_ALLOCS
		storage->blocks = realloc( storage->blocks, storage->blocks_count*sizeof(ASStorageBlock*));
#else
		storage->blocks = guarded_realloc( storage->blocks, storage->blocks_count*sizeof(ASStorageBlock*));
		show_debug( __FILE__,"select_storage_block",__LINE__,"reallocated %d blocks pointers", storage->blocks_count );
#endif		   
		UsedMemory += 16*sizeof(ASStorageBlock*) ;

		while( ++i < storage->blocks_count )
			storage->blocks[i] = NULL ;
	}	 
	storage->blocks[new_block] = create_asstorage_block( max(storage->default_block_size, compressed_size) );		
	if( storage->blocks[new_block] == NULL )  /* memory allocation failed ! */ 
		new_block = -1 ;
	return new_block+1;
}

static inline void
destroy_storage_slot( ASStorageBlock *block, int index )
{
	ASStorageSlot **slots = block->slots ;
	int i = index;
	
/*	if( i < block->first_free ) 
		block->first_free = i ;
 */
	slots[i] = NULL ; 
	if( block->last_used == index ) 
	{	
		while( --i > 0 ) 
		{	
			if( slots[i] != NULL ) 
				break;
			--(block->unused_count);
		}
		block->last_used = i<0?0:i;	 
	}else if( index < block->last_used ) 
		++(block->unused_count);
}

static inline void 
join_storage_slots( ASStorageBlock *block, ASStorageSlot *from_slot, ASStorageSlot *to_slot )
{
	ASStorageSlot *s, *next = AS_STORAGE_GetNextSlot(from_slot);

	from_slot->size = ASStorageSlot_USABLE_SIZE(from_slot);	
	do
	{
		s = next ;
		next = AS_STORAGE_GetNextSlot(s);
		from_slot->size += ASStorageSlot_FULL_SIZE(s) ;	
		LOCAL_DEBUG_OUT( "from  = %p, s = %p, next = %p, to = %p, from->size = %ld", from_slot, s, next, to_slot, from_slot->size );
		destroy_storage_slot( block, s->index );
	}while( s < to_slot );	
}


static inline void
defragment_storage_block( ASStorageBlock *block )
{
	ASStorageSlot *brk, *next_used, **slots = block->slots ;
	int i, first_free = -1;
	unsigned long total_free = 0 ;
	brk = next_used = block->start ; 
	
	
	for( i = 0 ; i <= block->last_used ; ++i ) 
	{
		if( slots[i] ) 
			if( slots[i]->flags == 0 ) 
				slots[i] = NULL ;
		if( slots[i] == NULL ) 
		{
			if( first_free < 0 ) 
				first_free = i ;
		}
	}
	while( --i > 0 ) 
		if( slots[i] != NULL ) 
			break;
	block->last_used = i ;
								
	while( brk < block->end ) 
	{	
		ASStorageSlot *used = next_used;
		while( used < block->end && used->flags == 0 ) 
			used = AS_STORAGE_GetNextSlot(used);
		LOCAL_DEBUG_OUT("brk = %p, used = %p, end = %p", brk, used, block->end );
		if( used >= block->end || next_used > block->end) 
		{
			total_free = (unsigned long)((CARD8*)block->end - (CARD8*)brk);
			if( total_free < ASStorageSlot_SIZE ) 
				total_free = 0 ;
			else
				total_free -= ASStorageSlot_SIZE ; 	
			break;
		}else
			next_used = AS_STORAGE_GetNextSlot(used);

		LOCAL_DEBUG_OUT("used = %p, used->size = %ld", used,used->size );
		if( next_used < block->end ) 
		{
			LOCAL_DEBUG_OUT("next_used = %p, next_used->size = %ld", 
							next_used, next_used->size );
		}
		if( used != brk	)
		{/* can't use memcpy as regions may overlap */
			int size = (ASStorageSlot_FULL_SIZE(used))/4;
			register CARD32 *from = (CARD32*)used ;
			register CARD32 *to = (CARD32*)brk ;
			for( i = 0 ; i < size ; ++i ) 
				to[i] = from[i];
		}	
		/* updating pointer : */	
		slots[brk->index] = brk ;
		LOCAL_DEBUG_OUT("brk = %p, brk->size = %ld, index = %d", brk, brk->size, brk->index );
		brk = AS_STORAGE_GetNextSlot(brk);
	}
	
	if( total_free > 0 )
	{
		if( first_free < 0  ) 
		{
			if( ++block->last_used >= block->slots_count ) 
				add_storage_slots( block );	
			first_free = block->last_used ;
		}
		brk->flags = 0 ;
		brk->size = total_free ; 
		brk->uncompressed_size = total_free ;
		brk->ref_count = 0 ;
		brk->index = first_free ; 
		block->first_free = first_free ;
		
		LOCAL_DEBUG_OUT("brk = %p, brk->size = %ld, index = %d, first_free = %d", 
						 brk, brk->size, brk->index, first_free );
		
		block->slots[first_free] = brk ;
		if( block->last_used < first_free ) 
			block->last_used = first_free ;
	}
	
	block->total_free = total_free ;
	LOCAL_DEBUG_OUT( "total_free after defrag = %ld, first_free = %d, last_used = %d", total_free, block->first_free, block->last_used );
	
	slots = block->slots ;
	for( i = 0 ; i <= block->last_used ; ++i ) 
		if( slots[i] )
			if( slots[i]->index != i ) 
			{
				LOCAL_DEBUG_OUT( "Storage Integrity check failed - block = %p, index = %d", block, i ) ;	
				exit(0);
			}
	block->unused_count = 0 ;
	for( i = 0 ; i < block->last_used ; ++i ) 
	{
		if( slots[i] == NULL ) 
			++(block->unused_count);
	}
}

static ASStorageSlot *
select_storage_slot( ASStorageBlock *block, int size )
{
	int i = block->first_free ;
	LOCAL_DEBUG_OUT( "first_free = %d, last_used = %d, long_searches = %d", block->first_free, block->last_used, block->long_searches );
	if( block->long_searches < 5 ) 
	{	
		int max_i = block->last_used ;
		ASStorageSlot **slots = block->slots ;
		int empty_slots_checked = 0 ;

		if( max_i > i + 50 ) 
			max_i = i+50 ; 
		
		while( i <= max_i )
		{
			ASStorageSlot *slot = slots[i] ;
			LOCAL_DEBUG_OUT( "block = %p, max_i = %d, slots[%d] = %p", block, max_i, i, slot );
			if( slot != NULL )
			{
				if( slot->flags == 0 )
				{	  
					int single_slot_size = size+ASStorageSlot_SIZE ; 
					int size_to_match = single_slot_size+ASStorageSlot_SIZE ;
					++empty_slots_checked ;
					
					do
					{
						ASStorageSlot *next_slot = AS_STORAGE_GetNextSlot(slot);		   
						if( next_slot > block->end )
							break;

						LOCAL_DEBUG_OUT( "start = %p, slot = %p, slot->size = %ld, end = %p, size = %d, size_to_match = %d", block->start, slot, slot->size, block->end, size, size_to_match );
						if((int)ASStorageSlot_USABLE_SIZE(slot) >= single_slot_size )
						{	
							if( empty_slots_checked > 50 ) ++(block->long_searches);
							return slot;
						}
						if( (int)ASStorageSlot_FULL_SIZE(slot)  >= size_to_match )
						{
							join_storage_slots( block, slots[i], slot );
							if( empty_slots_checked > 50 ) ++(block->long_searches);
							return slots[i];
						}	
						size_to_match -= ASStorageSlot_FULL_SIZE(slot);
						slot = next_slot;		
						/* make sure we has not exceeded boundaries of the block */									   
					}while( slot->flags == 0 );
				}
			}
			++i ;
		}
	}
		
	/* no free slots of sufficient size - need to do defragmentation */
	defragment_storage_block( block );
	block->long_searches = 0 ;
	i = block->first_free;
	if( i >= block->slots_count ) 
		return NULL;
    if( block->slots[i] == NULL || (int)block->slots[i]->size < size ) 
		return NULL;
	return block->slots[i];		   
}

static inline Bool
split_storage_slot( ASStorageBlock *block, ASStorageSlot *slot, int to_size )
{
	int old_size = ASStorageSlot_USABLE_SIZE(slot) ;
	ASStorageSlot *new_slot ;

	LOCAL_DEBUG_OUT( "slot->size = %ld", slot->size );
	
	slot->size = to_size ; 
	
	if( old_size <= (int)ASStorageSlot_USABLE_SIZE(slot) )
		return True;

	new_slot = AS_STORAGE_GetNextSlot(slot);

	LOCAL_DEBUG_OUT( "new_slot = %p, slot = %p, slot->size = %ld", new_slot, slot, slot->size );

	if( new_slot >=  block->end )
		return True;

	new_slot->flags = 0 ;
	new_slot->ref_count = 0 ;
	LOCAL_DEBUG_OUT( "old_size = %d, full_size = %ld", old_size, ASStorageSlot_FULL_SIZE(slot) );
	new_slot->size = old_size - ASStorageSlot_FULL_SIZE(slot) ;											   
	new_slot->uncompressed_size = 0 ;
	
	new_slot->index = 0 ;
	/* now we need to find where this slot's pointer we should store */		   
	if( block->unused_count < block->slots_count/10 && block->last_used < block->slots_count-1 )
	{	
		new_slot->index = ++(block->last_used) ;
	}else
	{
		register int i, max_i = block->slots_count ;
		register ASStorageSlot **slots = block->slots ;
		LOCAL_DEBUG_OUT( "max_i = %d", max_i );
		
		for( i = 0 ; i < max_i ; ++i ) 
			if( slots[i] == NULL ) 
				break;
		LOCAL_DEBUG_OUT( "i = %d", i );
		if( i >= max_i ) 
		{
			if( block->slots_count >= AS_STORAGE_MAX_SLOTS_CNT )
				return False;
			else
			{
				i = block->slots_count ;
				block->last_used = i ;
				add_storage_slots( block );
				slots = block->slots ;
			}	 
		}
		LOCAL_DEBUG_OUT( "i = %d", i );
 		new_slot->index = i ;		
		if( i < block->last_used )
		{
			if( block->unused_count <= 0 ) 
				show_warning( "Storage error : unused_count out of range (%d )", block->unused_count );
			else					  
				--(block->unused_count);
		}
	}	
	LOCAL_DEBUG_OUT( "new_slot = %p, new_slot->index = %d, new_slot->size = %ld", new_slot, new_slot->index, new_slot->size );
	block->slots[new_slot->index] = new_slot ;
	return True;
}

static int
store_data_in_block( ASStorageBlock *block, CARD8 *data, int size, int compressed_size, int ref_count, ASFlagType flags )
{
	ASStorageSlot *slot ;
	CARD8 *dst ;
	Bool bad_slot = True ;
	slot = select_storage_slot( block, compressed_size );
	LOCAL_DEBUG_OUT( "selected slot %p for size %d (compressed %d) and flags %lX", slot, size, compressed_size, flags );
	
	if( slot == NULL ) 
		return 0;           /* not a error condition */
	else if( slot > block->end || slot < block->start) 
		show_error( "storage slot selected falls outside of allocated memory. Slot = %p, start = %p, end = %p", slot, block->start, block->end );
	else if( &(ASStorage_Data(slot)[slot->size]) > ((CARD8*)(block->start)) + block->size) 
		show_error( "storage slot's size falls outside of allocated memory. Slot->data[slot->size] = %p, end = %p, size = %d", &(ASStorage_Data(slot)[slot->size]), ((CARD8*)(block->start)) + block->size, slot->size );
	else if( slot->index >= block->slots_count ) 
		show_error( "storage slot index falls out of range. Index = %d, slots_count = %d", slot->index, block->slots_count );
	else
		bad_slot = False ;
	
	if( bad_slot )
	{
		show_error( "\t data = %p, size = %d, compressed_size = %d, ref_count = %d, flags = 0x%lX", block, data, size, compressed_size, ref_count, flags	);
		show_error( "\t block = %p, : {size:%d, total_free:%d, slots_count:%d, unused_count:%d, first_free:%d, last_used:%d}", block, block->size, block->total_free, block->slots_count, block->unused_count, block->first_free, block->last_used );
		if( slot ) 
			show_error( "\t slot = %p : {flags:0x%X, ref_count:%u, size:%lu, uncompr_size:%lu, index:%u}", slot,
					 	slot->flags, slot->ref_count, slot->size, slot->uncompressed_size, slot->index );
		return 0;
	}			  
		
	LOCAL_DEBUG_OUT( "block = %p", block );
	if( !split_storage_slot( block, slot, compressed_size ) ) 
	{
		show_error( "failed to split storage to store data in block. Usable size = %d, desired size = %d", ASStorageSlot_USABLE_SIZE(slot), compressed_size+ASStorageSlot_SIZE );
		return 0 ;
	}
	LOCAL_DEBUG_OUT( "block = %p", block );
	block->total_free -= ASStorageSlot_FULL_SIZE(slot);
	
	dst = ASStorage_Data(slot);
	LOCAL_DEBUG_OUT( "dst = %p, compressed_size = %d", dst, compressed_size );
	memcpy( dst, data, compressed_size );
	slot->flags = (unsigned short)(flags | ASStorage_Used) ;
	slot->ref_count = ref_count;
	slot->size = compressed_size ;
	slot->uncompressed_size = size ;

	if( slot->index == block->first_free ) 
	{
		int i = block->first_free ;
		while( ++i <= block->last_used ) 
			if( block->slots[i] && 
				block->slots[i]->flags == 0 && block->slots[i]->size > 0 ) 
				break;
		block->first_free = i ;
	}

	LOCAL_DEBUG_OUT( "slot index = %d", slot->index );
 
	return slot->index+1 ;
}


static ASStorageID 
store_compressed_data( ASStorage *storage, CARD8* data, int size, int compressed_size, int ref_count, ASFlagType flags )
{
	int id = 0 ;
	int block_id = 0;
	
	do
	{	
		block_id = select_storage_block( storage, compressed_size, flags, block_id );
		LOCAL_DEBUG_OUT( "selected block %d", block_id );
		if( block_id > 0 ) 
		{
			int slot_id = store_data_in_block(  storage->blocks[block_id-1], 
												data, size, 
												compressed_size, ref_count, flags );

			LOCAL_DEBUG_OUT( "slot id %X", slot_id );
			if( slot_id > 0 )	
				id = make_asstorage_id( block_id, slot_id );
			else
				if( storage->blocks[block_id-1]->total_free >= compressed_size+ASStorageSlot_SIZE  ) 
				{
					show_error( "failed to store data in block. Total free size = %d, desired size = %d", storage->blocks[block_id-1]->total_free, compressed_size+ASStorageSlot_SIZE );
					break;
				}
		}
	}while( block_id != 0 && id == 0 );
	return id ;		
}	  




static inline ASStorageBlock *
find_storage_block( ASStorage *storage, ASStorageID id )
{	
	int block_idx = StorageID2BlockIdx(id);
	if( block_idx >= 0 && block_idx < storage->blocks_count )  
		return storage->blocks[block_idx];
	return NULL ;
}

static inline ASStorageSlot *
find_storage_slot( ASStorageBlock *block, ASStorageID id )
{	
	if( block != NULL ) 
	{
		int slot_idx = StorageID2SlotIdx(id);
		if( slot_idx >= 0 && slot_idx < block->slots_count ) 
		{
			if( block->slots[slot_idx] && block->slots[slot_idx]->flags != 0 )
				return block->slots[slot_idx];
		}
	}	
	return NULL ;
}

static inline void 
free_storage_slot( ASStorageBlock *block, ASStorageSlot *slot)
{
	slot->flags = 0 ;
	block->total_free += ASStorageSlot_USABLE_SIZE(slot) ;
}	 

static Bool 
is_block_empty( ASStorageBlock *block)
{
	int i = block->last_used+1;
	ASStorageSlot **slots = block->slots ;
	while( --i >= 0 ) 
	{
		if( slots[i] )
			if( slots[i]->flags != 0 ) 
				return False;	 
	}	
	return True;	
}	 

static void 
free_storage_block( ASStorage *storage, int block_idx  )
{
	ASStorageBlock *block = storage->blocks[block_idx] ;
	storage->blocks[block_idx] = NULL ;
	destroy_asstorage_block( block );
}	 

static ASStorageSlot *
convert_slot_to_ref( ASStorage *storage, ASStorageID id )	
{
	int block_idx = StorageID2BlockIdx(id);
	ASStorageBlock *block;
	ASStorageID target_id = 0;
	int slot_id = 0 ;
	int ref_index, body_index ;
	ASStorageSlot *ref_slot, *body_slot ;
	
	block = find_storage_block(storage, id);
	
	LOCAL_DEBUG_OUT( "block = %p, block->total_free = %d", block, block->total_free );
	/* Two strategies here - 1 - the fast one - we try to allocate new slot 
	 * and avoid copying the body of the data over - we can do that only if
	 * there is enough space in its block, otherwise we have to relocate it 
	 * into different block, which is slower.
	 */
	if( block->total_free > sizeof(ASStorageID))
	{	
		slot_id = store_data_in_block(  block, (CARD8*)&target_id, 
										sizeof(ASStorageID), sizeof(ASStorageID), 0, 
										ASStorage_Reference );
	}
	LOCAL_DEBUG_OUT( "block = %p, block->total_free = %d, slot_id = 0x%X", block, block->total_free, slot_id );
	
	if( slot_id > 0 )
	{ 	/* We can use fast strategy : now we need to swap contents of the slots */
		ref_index = slot_id-1 ;
		ref_slot = block->slots[ref_index] ;
	
		body_index = StorageID2SlotIdx(id) ; 
		body_slot = block->slots[body_index] ;
	
		block->slots[ref_index] = body_slot ;
		body_slot->index = ref_index ;

		block->slots[body_index] = ref_slot ; 
		ref_slot->index = body_index ;

		target_id = make_asstorage_id( block_idx+1, slot_id );
		if( target_id == id ) 
		{
			show_error( "Reference ID is the same as target_id: id = %lX, slot_id = %d", id, slot_id );
#ifndef NO_DEBUG_OUTPUT
			{	int *a = NULL ; *a = 0 ;}
#endif						   
		}
		/* don't increment refcount, becouse we oonly have one published reference to it so far */
		/* ++(body_slot->ref_count); */
	}else
	{/* otherwise we have to relocate the actuall body into a different block, 
	  * which is somewhat tricky : */
		ref_index = StorageID2SlotIdx(id); ;
		ref_slot = block->slots[ref_index] ;
		
		if( block->total_free > (int)ref_slot->size )
		{
			/* there is a danger of us trying to reuse same block and defragmented it in between,
			 * which will screw up the data */
#ifndef NO_DEBUG_OUTPUT
			fprintf( stderr, "\t\t %s:%d DANGEROUS RELOCATION! size = %ld",  __FILE__, __LINE__, ref_slot->size );
#endif						   
			memcpy( storage->comp_buf, ASStorage_Data(ref_slot), ref_slot->size );
			target_id = store_compressed_data(  storage, storage->comp_buf, 
										   		ref_slot->uncompressed_size, 
										   		ref_slot->size, ref_slot->ref_count, ref_slot->flags );
		}else	 
			target_id = store_compressed_data( storage, ASStorage_Data(ref_slot), 
										   	ref_slot->uncompressed_size, 
										   	ref_slot->size, ref_slot->ref_count, ref_slot->flags );
		/* lets do this again, in case block was defragmented */
		ref_slot = block->slots[ref_index] ;

		if( target_id == 0 ) 
			return NULL;		
		if( target_id == id ) 
		{	
			show_error( "Reference ID is the same as target_id: id = %lX" );
#ifndef NO_DEBUG_OUTPUT
			{	int *a = NULL ; *a = 0 ;}
#endif						   
		}
		
		split_storage_slot( block, ref_slot, sizeof(ASStorageID));
		ref_slot->uncompressed_size = sizeof(ASStorageID) ; 
		set_flags( ref_slot->flags, ASStorage_Reference );
		clear_flags( ref_slot->flags, ASStorage_CompressionType );
	}	 
	memcpy( ASStorage_Data(ref_slot), (CARD8*)&target_id, sizeof(ASStorageID));				 

	return ref_slot;
}

typedef struct 
{
	int offset ; 
	void *buffer ;
	
	unsigned int threshold ;
	int start, end, runs_count ;
}ASStorageDstBuffer;

typedef void (*data_cpy_func_type)(ASStorageDstBuffer *, void *, size_t);

static void card8_card8_cpy( ASStorageDstBuffer *dst, void *src, size_t size)
{
	register CARD8 *dst8 = (CARD8*)dst->buffer ;
	dst8 += dst->offset ;
	memcpy( dst8, src, size );
}	 


static void card8_card32_cpy( ASStorageDstBuffer *dst, void *src, size_t size)
{
	register CARD32 *dst32 = (CARD32*)dst->buffer + dst->offset ;
	register CARD8  *src8  = (CARD8*)src ;
	register int i;
	for( i = 0 ;  i < (int)size ; ++i ) 
		dst32[i] = src8[i] ;
}	 

static void 
card8_threshold( ASStorageDstBuffer *dst, void *src, size_t size)
{
	CARD8 *src8 = src ;
	unsigned int *runs = (unsigned int*)(dst->buffer) ;
	int runs_count = dst->runs_count ;
	unsigned int threshold = dst->threshold ;
	int start = dst->start, end = dst->end;
	int i = 0;

#ifdef DEBUG_THRESHOLD	  
	fprintf( stderr, "card8_threshold:enter: start = %d, end = %d, runs_count = %d, size = %d\n", 
			 start, end, runs_count, size );
#endif

	while( i < (int)size ) 
	{
		if( end < start ) 
		{
			while( i < (int)size && src8[i] < threshold ) 
				++i ;
			start = i ;
		}	 
#ifdef DEBUG_THRESHOLD	  
		fprintf( stderr, "card8_threshold:1: start = %d, end = %d, i = %d\n", start, end, i );
#endif
		
		if( i < (int)size ) 
		{	
			while( i < (int)size && src8[i] >= threshold ) 
				++i ;
			end = i-1 ;
		}
#ifdef DEBUG_THRESHOLD	  
		fprintf( stderr, "card8_threshold:2: start = %d, end = %d, i = %d\n", start, end, i );
#endif
		
		if( start >= 0 && end >= start )
		{
			runs[runs_count] = start ;
			++runs_count;
			runs[runs_count] = end ;
			++runs_count ;
#ifdef DEBUG_THRESHOLD	  
			fprintf( stderr, "card8_threshold:3: runs_count = %d\n", runs_count );
#endif
			end = -1 ;
		}
	}
#ifdef DEBUG_THRESHOLD	  
	fprintf( stderr, "card8_threshold:exit: start = %d, end = %d, runs_count = %d, size = %d\n", 
			 start, end, runs_count, size );
#endif
	dst->runs_count = runs_count ;
	dst->start = start ; 
	dst->end = end ;
}	 

static int  
fetch_data_int( ASStorage *storage, ASStorageID id, ASStorageDstBuffer *buffer, int offset, int buf_size, CARD8 bitmap_value, 
		  		data_cpy_func_type cpy_func, int *original_size)
{
	ASStorageSlot *slot = find_storage_slot( find_storage_block( storage, id ), id );
	LOCAL_DEBUG_OUT( "slot = %p", slot );
	if( slot && buffer && buf_size > 0 )
	{
		int uncomp_size = slot->uncompressed_size ;
		*original_size = uncomp_size ;
		if( get_flags( slot->flags, ASStorage_Reference) )
		{
			ASStorageID target_id = 0;
			memcpy( &target_id, ASStorage_Data(slot), sizeof( ASStorageID ));				   
			LOCAL_DEBUG_OUT( "target_id = %lX", target_id );
			if( target_id != 0 ) 
				return fetch_data_int(storage, target_id, buffer, offset, buf_size, bitmap_value, cpy_func, original_size);
			else
				return 0;
		}	 

		LOCAL_DEBUG_OUT( "flags = %X, index = %d, size = %ld, uncompressed_size = %d", 
							slot->flags, slot->index, slot->size, uncomp_size );
		if( bitmap_value == 0 ) 
			bitmap_value = AS_STORAGE_DEFAULT_BMAP_VALUE ;

		{
			CARD8 *tmp = decompress_stored_data( storage, ASStorage_Data(slot), slot->size,
													uncomp_size, slot->flags, bitmap_value );
			while( offset > uncomp_size ) offset -= uncomp_size ; 
			while( offset < 0 ) offset += uncomp_size ; 
			
			if( get_flags( slot->flags, ASStorage_NotTileable ) )
				if( buf_size > uncomp_size - offset ) 
					buf_size = uncomp_size - offset ;
			if( offset > 0 ) 
			{
				int to_copy = uncomp_size-offset ; 
				if( to_copy > buf_size ) 
					to_copy = buf_size ;
				cpy_func( buffer, tmp+offset, to_copy ); 															
				buffer->offset = to_copy ;
			}
			LOCAL_DEBUG_OUT( "offset = %d", buffer->offset );
			while( buffer->offset < buf_size ) 
			{
				int to_copy = buf_size - buffer->offset ; 
				if( to_copy > uncomp_size ) 
					to_copy = uncomp_size ;
				cpy_func( buffer, tmp, to_copy ); 															
				buffer->offset += to_copy;
			}
		}
		LOCAL_DEBUG_OUT( "uncompressed_size = %d", buffer->offset );
		return buffer->offset ;
	}
	return 0;
}

/************************************************************************/
/* Public Functions : 													*/
/************************************************************************/
ASStorage *
create_asstorage()
{
#ifndef DEBUG_ALLOCS
	ASStorage *storage = calloc(1, sizeof(ASStorage));
#else
	ASStorage *storage = guarded_calloc(1, sizeof(ASStorage));
#endif
	UsedMemory += sizeof(ASStorage) ;
	if( storage )
		storage->default_block_size = AS_STORAGE_DEF_BLOCK_SIZE ;
	return storage ;
}

int 
set_asstorage_block_size( ASStorage *storage, int new_size )
{
	int old_size ;
	
	if( storage == NULL ) 
		storage = get_default_asstorage();
	
	old_size = storage->default_block_size ; 
#if 1
	if( new_size > AS_STORAGE_DEF_BLOCK_SIZE ) 
		storage->default_block_size = new_size; 
	else
		storage->default_block_size = AS_STORAGE_DEF_BLOCK_SIZE; 
#endif
	return old_size;
}

void 
destroy_asstorage(ASStorage **pstorage)
{
	ASStorage *storage = *pstorage ;
	
	if( storage ) 
	{	
		if( storage->blocks != NULL && storage->blocks_count  > 0 )
		{
			int i ;
			for( i = 0 ; i < storage->blocks_count ; ++i ) 
				if( storage->blocks[i] ) 
					destroy_asstorage_block( storage->blocks[i] );
			UsedMemory -= storage->blocks_count * sizeof(ASStorageBlock*) ;
#ifndef DEBUG_ALLOCS
			free( storage->blocks );
#else	
			guarded_free( storage->blocks );
#endif

		}	
		if( storage->comp_buf )
			free( storage->comp_buf);
		if( storage->diff_buf )
			free( storage->diff_buf);

		UsedMemory -= sizeof(ASStorage) ;
#ifndef DEBUG_ALLOCS
		free( storage );
#else	
		guarded_free( storage );
#endif
		*pstorage = NULL;
	}
}

void 
flush_default_asstorage()
{
	if( _as_default_storage != NULL )
		destroy_asstorage(&_as_default_storage);
}

ASStorageID 
store_data(ASStorage *storage, CARD8 *data, int size, ASFlagType flags, CARD8 bitmap_threshold)
{
	int compressed_size = size ;
	CARD8 *buffer = data;
	CARD32 bitmap_threshold32 = bitmap_threshold ;

	if( storage == NULL ) 
		storage = get_default_asstorage();

	LOCAL_DEBUG_CALLER_OUT( "data = %p, size = %d, flags = %lX", data, size, flags );
	if( size <= 0 || data == NULL || storage == NULL ) 
		return 0;
	if( get_flags( flags, ASStorage_Bitmap ) )
	{
		if( bitmap_threshold32 == 0 ) 
			bitmap_threshold32 = AS_STORAGE_DEFAULT_BMAP_THRESHOLD ;
	}else
		bitmap_threshold32 = 0x000000FF ;  /* to disable the tint ! */ 
			 
	if( !get_flags(flags, ASStorage_Reference))
		if( get_flags( flags, ASStorage_CompressionType ) || get_flags( flags, ASStorage_32Bit ) )
			buffer = compress_stored_data( storage, data, size, &flags, &compressed_size, bitmap_threshold32 );
	
	return store_compressed_data( storage, buffer, 
								  get_flags( flags, ASStorage_32Bit )?size/4:size, 
								  compressed_size, 0, flags );
}

ASStorageID 
store_data_tinted(ASStorage *storage, CARD8 *data, int size, ASFlagType flags, CARD16 tint)
{
	int compressed_size = size ;
	CARD8 *buffer = data;
	CARD32 tint32 = tint ;

	if( storage == NULL ) 
		storage = get_default_asstorage();

	LOCAL_DEBUG_CALLER_OUT( "data = %p, size = %d, flags = %lX", data, size, flags );
	if( size <= 0 || data == NULL || storage == NULL ) 
		return 0;
	
	if( get_flags( flags, ASStorage_Bitmap ) )
	{
		if( tint32 == 0 ) 
			tint32 = 0x000000FF ;
		else
			tint32 = (tint32 * AS_STORAGE_DEFAULT_BMAP_THRESHOLD) >>8 ;
	}
	
	if( !get_flags(flags, ASStorage_Reference))
		if( get_flags( flags, ASStorage_CompressionType ) || get_flags( flags, ASStorage_32Bit ) )
			buffer = compress_stored_data( storage, data, size, &flags, &compressed_size, tint32 );
	
	return store_compressed_data( storage, buffer, 
								  get_flags( flags, ASStorage_32Bit )?size/4:size, 
								  compressed_size, 0, flags );
}


int  
fetch_data(ASStorage *storage, ASStorageID id, CARD8 *buffer, int offset, int buf_size, CARD8 bitmap_value, int *original_size)
{
	int dumm ; 
	if( storage == NULL ) 
		storage = get_default_asstorage();

	if( original_size == NULL ) 
		original_size = &dumm ;
	*original_size = 0;
	if( storage != NULL && id != 0 )
	{	
		ASStorageDstBuffer buf ; 
		buf.offset = 0 ; 
		buf.buffer = buffer ;
		return fetch_data_int( storage, id, &buf, offset, buf_size, bitmap_value, card8_card8_cpy, original_size );
	}
	return 0 ;	 
}

int  
fetch_data32(ASStorage *storage, ASStorageID id, CARD32 *buffer, int offset, int buf_size, CARD8 bitmap_value, int *original_size)
{
	int dumm ;
	if( storage == NULL ) 
		storage = get_default_asstorage();
	
	if( original_size == NULL ) 
		original_size = &dumm ;
	*original_size = 0;
	if( storage != NULL && id != 0 )
	{
		ASStorageDstBuffer buf ; 
		buf.offset = 0 ; 
		buf.buffer = buffer ;
	  	
		return fetch_data_int( storage, id, &buf, offset, buf_size, bitmap_value, card8_card32_cpy, original_size );
	}
	return 0 ;	
}

int  
threshold_stored_data(ASStorage *storage, ASStorageID id, unsigned int *runs, int width, unsigned int threshold)
{
	if( storage == NULL ) 
		storage = get_default_asstorage();
	
	if( storage != NULL && id != 0 )
	{
		ASStorageDstBuffer buf ; 
		int dumm = 0 ;
		buf.offset = 0 ; 
		buf.buffer = runs ;

		buf.threshold = threshold ; 
		buf.start = 0 ;
		buf.end = -1 ;
		buf.runs_count = 0 ;
#ifdef DEBUG_THRESHOLD	  
		fprintf( stderr, "threshold_stored_data: id = 0x%lX, width = %d, threshold = %d\n", id, width, threshold );
#endif
		if( fetch_data_int( storage, id, &buf, 0, width, (CARD8)threshold, card8_threshold, &dumm) > 0 ) 
		{
			if( buf.start >= 0 && buf.end >= buf.start )
			{
				runs[buf.runs_count] = buf.start ;
				++buf.runs_count;
				runs[buf.runs_count] = buf.end ;
				++buf.runs_count ;
			}	 
			return buf.runs_count;
		}
	}
	return 0 ;	
}


Bool 
query_storage_slot(ASStorage *storage, ASStorageID id, ASStorageSlot *dst )
{
	if( storage == NULL ) 
		storage = get_default_asstorage();
	
	if( storage != NULL && id != 0 && dst != NULL )
	{	
		ASStorageSlot *slot = find_storage_slot( find_storage_block( storage, id ), id );
		LOCAL_DEBUG_OUT( "slot = %p", slot );
		if( slot )
		{
			if( get_flags( slot->flags, ASStorage_Reference) )
			{
				ASStorageID target_id = 0;
			 	memcpy( &target_id, ASStorage_Data(slot), sizeof( ASStorageID ));				   
				LOCAL_DEBUG_OUT( "target_id = %lX", target_id );
				if( target_id == id ) 
				{
					show_error( "reference refering to self id = %lX", id );
					return False;
				}
				return query_storage_slot(storage, target_id, dst);
			}	 
			*dst = *slot ;
			return True ;
		}
	}
	return False;	  
}

int 
print_storage_slot(ASStorage *storage, ASStorageID id)
{
	if( storage == NULL ) 
		storage = get_default_asstorage();
	
	if( storage != NULL && id != 0 )
	{	
		ASStorageSlot *slot = find_storage_slot( find_storage_block( storage, id ), id );
		fprintf (stderr, "Storage ID 0x%lX-> slot %p", (unsigned long)id, slot);
		if( slot )
		{
			int i ;
			if( get_flags( slot->flags, ASStorage_Reference) )
			{
				ASStorageID target_id = 0;
			 	memcpy( &target_id, ASStorage_Data(slot), sizeof( ASStorageID ));				   
				fprintf (stderr, " : References storage ID 0x%lX\n\t>", (unsigned long)target_id);
				if( target_id == id ) 
				{	
					show_error( "reference refering to self id = %lX", id );
					return 0;
				}
				return print_storage_slot(storage, target_id);
			}	 
			fprintf( stderr, " : {0x%X, %u, %lu, %lu, %u, {", 
					 slot->flags, slot->ref_count, (unsigned long)slot->size, (unsigned long)slot->uncompressed_size, slot->index );

			for( i = 0 ; i < (int)slot->size ; ++i)
				fprintf( stderr, "%2.2X ", ASStorage_Data(slot)[i] ) ;
			fprintf (stderr, "}}");
			return slot->size + ASStorageSlot_SIZE ;
		}
		fprintf (stderr, "\n");
	}
	return 0;	  
}	 

void  
print_storage(ASStorage *storage)
{
	int i ;
	if( storage == NULL ) 
		storage = get_default_asstorage();
	fprintf( stderr, " Printing Storage %p : \n\tblock_count = %d;\n", storage, storage->blocks_count );

	for( i = 0 ; i < storage->blocks_count ; ++i ) 
	{
		fprintf( stderr, "\tBlock %d = %p;\n", i, storage->blocks[i] );			
		if( storage->blocks[i] )
		{
			fprintf( stderr, "\t\tBlock[%d].size = %d;\n", i, storage->blocks[i]->size );			   
			fprintf( stderr, "\t\tBlock[%d].slots_count = %d;\n", i, storage->blocks[i]->slots_count );			   
			fprintf( stderr, "\t\tBlock[%d].last_used = %d;\n", i, storage->blocks[i]->last_used );			   
		}	 
	}	 
}

void 
forget_data(ASStorage *storage, ASStorageID id)
{
	if( storage == NULL ) 
		storage = get_default_asstorage();
	
	if( storage != NULL && id != 0 ) 
	{
		ASStorageBlock *block = find_storage_block( storage, id );
 		ASStorageSlot  *slot  = find_storage_slot( block, id );				
		if( block && slot ) 
		{
			if( get_flags( slot->flags, ASStorage_Reference) )
			{
				ASStorageID target_id = 0;
			 	memcpy( &target_id, ASStorage_Data(slot), sizeof( ASStorageID ));				   
				if( target_id != id ) 
					forget_data( storage, target_id );					
				else
					show_error( "reference refering to self id = %lX", id );
			}	 
			LOCAL_DEBUG_OUT( "id = %lX, ref_count = %d;", id, slot->ref_count );
			if( slot->ref_count >= 1 ) 
				--(slot->ref_count);
			else
			{	
				free_storage_slot(block, slot);
				if( is_block_empty(block) ) 
					free_storage_block( storage, StorageID2BlockIdx(id) );
			}
		}	 
	}			  
}

ASStorageID 
dup_data(ASStorage *storage, ASStorageID id)
{
	ASStorageID new_id = 0 ;

	if( storage == NULL ) 
		storage = get_default_asstorage();
	   
	if( storage != NULL && id != 0 )
	{	
		ASStorageSlot *slot = find_storage_slot( find_storage_block( storage, id ), id );
		LOCAL_DEBUG_OUT( "slot = %p, slot->index = %d, index(id) = %ld", slot, slot?slot->index:-1, StorageID2SlotIdx(id) );
		if( slot )
		{
			ASStorageSlot *target_slot = NULL;
			ASStorageID target_id = id ;
			if( !get_flags( slot->flags, ASStorage_Reference )) 
			{	
				ASStorageSlot *new_slot = convert_slot_to_ref( storage, id );
				if( new_slot != NULL ) 
					slot = new_slot;
			}
				
			if( get_flags( slot->flags, ASStorage_Reference )) 
			{   
				memcpy( &target_id, ASStorage_Data(slot), sizeof( ASStorageID ));
				/* from now on - slot is a reference slot, so we just need to 
			 	 * duplicate it and increase ref_count of target */
				if( target_id != id ) 
					target_slot = find_storage_slot( find_storage_block( storage, target_id ), target_id );
				else
					show_error( "reference refering to self id = %lX", id );
			}else
				target_slot = slot ; 	
			
			LOCAL_DEBUG_OUT( "target_slot = %p, slot = %p", target_slot, slot );
			if( target_slot == NULL ) 
				return 0;
			/* doing it here as store_data() may change slot pointers */
			++(target_slot->ref_count);			   
			new_id = store_data( storage, (CARD8*)&target_id, sizeof(ASStorageID), ASStorage_Reference, 0);
			LOCAL_DEBUG_OUT( "new_id = 0x%lX, target_id = %lX, target->ref_count = %d", new_id, target_id, target_slot->ref_count );
		}
	}
	return new_id;
}

/*************************************************************************/
/* test code */
/*************************************************************************/
#ifdef TEST_ASSTORAGE
#include "afterimage.h"

#define STORAGE_TEST_KINDS	7
static int StorageTestKinds[STORAGE_TEST_KINDS][2] = 
{
	{10, 10000 },
	{100, 5000 },
	{4096, 5000 },
	{128*1024, 128 },
	{256*1024, 32 },
	{512*1024, 16 },
	{1024*1024, 8 }
};	 

CARD8 Buffer[1024*1024] ;
/* #define STORAGE_TEST_COUNT  1 */
#define STORAGE_TEST_COUNT  8+16+32+128+5000+5000+10000
typedef struct ASStorageTest {
	int size ;
	CARD8 *data;
	Bool linked ;
	ASStorageID id ;
}ASStorageTest;
 
static ASStorageTest Tests[STORAGE_TEST_COUNT];

static ASImageDecoder *imdec = NULL ;

void
make_storage_test_data( ASStorageTest *test, int min_size, int max_size, ASFlagType flags )
{
	int size = random()%max_size ;
	int i ;
	static CARD32 rnd32_seed = 345824357;
	static int chan = 0 ;
	CARD32 *data ;
	CARD8 *test_data8 ;
	CARD32 *test_data32 ;

#define MAX_MY_RND32		0x00ffffffff
#ifdef WORD64
#define MY_RND32() \
(rnd32_seed = ((1664525L*rnd32_seed)&MAX_MY_RND32)+1013904223L)
#else
#define MY_RND32() \
(rnd32_seed = (1664525L*rnd32_seed)+1013904223L)
#endif
	
	if( size <= min_size )
		size += min_size ;
	if( get_flags( flags, ASStorage_32Bit ) ) 	
		size = ((size/4)+1)*4 ;
	test->size = size ; 	   
#ifndef DEBUG_ALLOCS
	test->data = malloc(size);
#else
	test->data = guarded_malloc(size);
#endif

	test_data32	= (CARD32*)(test->data);
	test_data8	= test->data ;

	if( get_flags( flags, ASStorage_32Bit ) )
		size = size / 4 ;

	test->linked = False ;
	

	if( imdec ) 
	{	
		int k = 0;
		if( chan == 0 ) 
			imdec->decode_image_scanline( imdec );
		data = imdec->buffer.channels[chan];
		++chan ; 
		if( chan >= 3 ) 
			chan = 0 ;
		if( get_flags( flags, ASStorage_32Bit ) )
		{	
			if( get_flags( flags, ASStorage_8BitShift ) )
			{
				for( i = 0 ; i < size ; ++i ) 
				{	
					test_data32[i] = ((CARD32)data[k])<<8 ;
					if( ++k >= imdec->im->width )		k = 0 ;
				}
			}else	 
				for( i = 0 ; i < size ; ++i ) 
				{	
					test_data32[i] = data[k] ;
					if( ++k >= imdec->im->width )		k = 0 ;
				}
		}else
			for( i = 0 ; i < size ; ++i ) 
			{	
				test_data8[i] = data[k] ;
				if( ++k >= imdec->im->width )		k = 0 ;
			}
			
	}else
	{	
		if( get_flags( flags, ASStorage_32Bit ) )
		{	
			if( get_flags( flags, ASStorage_8BitShift ) )
			{
				for( i = 0 ; i < size ; ++i ) 
					test_data32[i] = (MY_RND32())&0x0000FF00 ;
			}else	 
				for( i = 0 ; i < size ; ++i ) 
					test_data32[i] = (MY_RND32())&0x000000FF ;
		}else
			for( i = 0 ; i < size ; ++i ) 
				test_data8[i] = MY_RND32() ;
	}
	test->id = 0 ;
}

int 
test_data_integrity( CARD8 *a, CARD8* b, int size, ASFlagType flags ) 
{
	register int i ;
	CARD32 *b32 = (CARD32*)b;
	CARD32 threshold32 = AS_STORAGE_DEFAULT_BMAP_THRESHOLD ;
	CARD32 threshold8 = AS_STORAGE_DEFAULT_BMAP_THRESHOLD ;

	if( get_flags( flags, ASStorage_32Bit ) )
	{	
		size = size / 4 ;
		if( get_flags( flags, ASStorage_8BitShift ) )
			threshold32 = threshold32 << 8 ;				
	}

	for( i = 0 ; i < size ; ++i ) 
	{
		Bool fail = False ;
		
		if( get_flags( flags, ASStorage_Bitmap ) )
		{	
			if( get_flags( flags, ASStorage_32Bit ) )
				fail = ( (a[i] >  threshold8 && b32[i] <= threshold32 )||(a[i] <= threshold8 && b32[i] > threshold32));
			else
				fail = ( (a[i] >  threshold8 && b[i] <= threshold8 )||(a[i] <= threshold8 && b[i] >  threshold8));

		}else
		{
			if( get_flags( flags, ASStorage_32Bit ) )
			{	
				if( get_flags( flags, ASStorage_8BitShift ) )
				{
					fail = ( (CARD8)(b32[i]>>8) != a[i] );
				}else
					fail = ( (CARD8)b32[i] != a[i] );
			}else		
				fail = ( a[i] != b[i] );
		}		
		if( fail ) 
		{
			int k ;
			if( get_flags( flags, ASStorage_32Bit ) )
			{	
				fprintf( stderr, "\tBytes %d differ : a[%d] == 0x%2.2X, b32[%d] == 0x%8.8lX\na: ", i, i, a[i], i, b32[i] );
				for( k = 0 ; k < size ; ++k ) 
					fprintf( stderr, (k==i)?"##%8.8X## ":"%8.8X ", a[k] );
				fprintf( stderr, "\nb: " );
				for( k = 0 ; k < size ; ++k ) 
					fprintf( stderr, (k==i)?"##%8.8lX## ":"%8.8lX ", b32[k] );
			
			}else
			{	
				fprintf( stderr, "\tBytes %d differ : a[%d] == 0x%2.2X, b[%d] == 0x%2.2X\na: ", i, i, a[i], i, b[i] );
				for( k = 0 ; k < size ; ++k ) 
					fprintf( stderr, (k==i)?"##%2.2X## ":"%2.2X ", a[k] );
				fprintf( stderr, "\nb: " );
				for( k = 0 ; k < size ; ++k ) 
					fprintf( stderr, (k==i)?"##%2.2X## ":"%2.2X ", b[k] );
			}

			fprintf( stderr, "\n" );
			return 1;			
		}			   
	}
	return 0 ;
}
	  
Bool 
test_asstorage(Bool interactive, int all_test_count, ASFlagType test_flags )
{
	ASStorage *storage ;
	ASStorageID id ;
	int i, kind, k;
	int min_size, max_size ;
	int test_count ;
	START_TIME(started);

	UsedMemory = 0 ;
	UncompressedSize = 0 ;
	CompressedSize = 0 ;

	fprintf( stderr, "\n%d :Testing flags 0x%lX @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n", __LINE__, test_flags );
	fprintf(stderr, "Testing storage creation ...");
	storage = create_asstorage();
#define TEST_EVAL(val)   do{ \
							if(!(val)){ fprintf(stderr, "failed\n"); return 1;} \
							else fprintf(stderr, "success.\n");}while(0)
	TEST_EVAL( storage != NULL ); 
	
#ifdef DO_CLOCKING	
	fprintf(stderr, "Testing speed for flags 0x%lX...", test_flags);
	{
#define SPEED_SIZE 1024*1024*50		
		CARD8 *speed_buffer ;
		int i, id ;
		static CARD32 rnd32_seed = 345824357;

		speed_buffer = safemalloc( SPEED_SIZE );

		for( i = 0 ; i < SPEED_SIZE ; ++i ) 
			speed_buffer[i] = (MY_RND32())&0x00FF ;
		
		{
			START_TIME(started2);
			id = store_data( storage, &speed_buffer[0], SPEED_SIZE, test_flags, 0 );
			SHOW_TIME("RLE compression speed ", started2);
		}
		{
			START_TIME(started3);
			fetch_data(storage, id, &speed_buffer[0], 0, SPEED_SIZE, 0, NULL);
			SHOW_TIME("RLE de-compression speed ", started3);
			forget_data(storage, id );
		}
		free( speed_buffer );
	}	 

#endif
	
	fprintf(stderr, "Testing store_data for data %p size = %d, and flags 0x%lX...", NULL, 0,
			test_flags);
	id = store_data( storage, NULL, 0, test_flags, 0 );
	TEST_EVAL( id == 0 ); 

	kind = 0 ; 
	min_size = 1 ;
	max_size = StorageTestKinds[kind][0] ; 
	test_count = StorageTestKinds[kind][1] ;
	for( i = 0 ; i < all_test_count ; ++i ) 
	{
		make_storage_test_data( &(Tests[i]), min_size, max_size, test_flags );
		fprintf(stderr, "Testing store_data for data %p size = %d, and flags 0x%lX...", Tests[i].data, Tests[i].size,
				test_flags);
		Tests[i].id = store_data( storage, Tests[i].data, Tests[i].size, test_flags, 0 );
		TEST_EVAL( Tests[i].id != 0 ); 
		fprintf(stderr, "\tstored with id = %lX...\n", Tests[i].id );

		if( --test_count <= 0 )
		{
			if( ++kind >= all_test_count ) 
				break;
			min_size = max_size ;
			max_size = StorageTestKinds[kind][0] ; 
			test_count = StorageTestKinds[kind][1] ;
		}		   
	}	 

	fprintf( stderr, "%d :memory used %d #####################################################\n", __LINE__, UsedMemory );
	fprintf( stderr, "%d :compressed_size = %d, uncompressed_size = %d, ratio = %d %% ###########\n", __LINE__, CompressedSize, UncompressedSize, (UncompressedSize<100)?0:(CompressedSize/(UncompressedSize/100)) );
	SHOW_TIME("Pass 1", started);

	if( interactive )
	   fgetc(stdin);
	for( i = 0 ; i < all_test_count ; ++i ) 
	{
		int size ;
		int res ;
		fprintf(stderr, "Testing fetch_data for id %lX size = %d ...", Tests[i].id, Tests[i].size);
		size = fetch_data(storage, Tests[i].id, &(Buffer[0]), 0, Tests[i].size, 0, NULL);
		TEST_EVAL( size == Tests[i].size ); 
		
		fprintf(stderr, "Testing fetched data integrity ...");
		res = test_data_integrity( &(Buffer[0]), Tests[i].data, size, test_flags );
		TEST_EVAL( res == 0 ); 
	}	 
	if( !get_flags( test_flags, ASStorage_32Bit ) )
	{	
		for( i = 0 ; i < all_test_count ; ++i ) 
		{
			int size ;
			int res ;
			fprintf(stderr, "Testing fetch_data32 for id %lX size = %d ...", Tests[i].id, Tests[i].size);
			size = fetch_data32(storage, Tests[i].id, &(Buffer[0]), 0, Tests[i].size/4, 0, NULL);
			TEST_EVAL( size == Tests[i].size/4 ); 
		
			fprintf(stderr, "Testing fetched data integrity ...");
			res = test_data_integrity( Tests[i].data, &(Buffer[0]), size, test_flags|ASStorage_32Bit );
			TEST_EVAL( res == 0 ); 
		}	 
	}

	fprintf( stderr, "%d :memory used %d #####################################################\n", __LINE__, UsedMemory );
	SHOW_TIME("Pass 2", started);
	if( interactive )
	   fgetc(stdin);
	for( i = 0 ; i < all_test_count ; ++i ) 
	{
		int size ;
		int r = random();
		if( (r&0x01) == 0 || Tests[i].id == 0 ) 
			continue;
		fprintf(stderr, "%d: Testing forget_data for id %lX size = %d ...\n", __LINE__, Tests[i].id, Tests[i].size);
		forget_data(storage, Tests[i].id);
		size = fetch_data(storage, Tests[i].id, &(Buffer[0]), 0, Tests[i].size, 0, NULL );
		TEST_EVAL( size != Tests[i].size ); 
		Tests[i].id = 0;
#ifndef DEBUG_ALLOCS
		free( Tests[i].data );
#else
		guarded_free( Tests[i].data );
#endif
		Tests[i].data = NULL ; 
		Tests[i].size = 0 ;
	}	 
	fprintf( stderr, "%d :memory used %d #####################################################\n", __LINE__, UsedMemory );
	SHOW_TIME("Pass 3", started);
	if( interactive )
	   fgetc(stdin);
	kind = 0 ; 
	min_size = 1 ;
	max_size = StorageTestKinds[kind][0] ; 
	test_count = StorageTestKinds[kind][1] ;
	for( i = 0 ; i < all_test_count ; ++i ) 
	{
		if( Tests[i].id == 0 ) 
		{	
			make_storage_test_data( &(Tests[i]), min_size, max_size, test_flags );
			fprintf(stderr, "Testing store_data for data %p size = %d, and flags 0x%lX...\n", Tests[i].data, Tests[i].size,
					test_flags);
			Tests[i].id = store_data( storage, Tests[i].data, Tests[i].size, test_flags, 0 );
			TEST_EVAL( Tests[i].id != 0 ); 
		}
		if( --test_count <= 0 )
		{
			if( ++kind >= all_test_count ) 
				break;
			min_size = max_size ;
			max_size = StorageTestKinds[kind][0] ; 
			test_count = StorageTestKinds[kind][1] ;
		}		   
	}	 
	fprintf( stderr, "%d :memory used %d #####################################################\n", __LINE__, UsedMemory );
	SHOW_TIME("Pass 4", started);
	if( interactive )
	   fgetc(stdin);
	for( i = 0 ; i < all_test_count ; ++i ) 
	{
		int size ;
		int res ;
		fprintf(stderr, "Testing fetch_data for id %lX size = %d ...", Tests[i].id, Tests[i].size);
		size = fetch_data(storage, Tests[i].id, &(Buffer[0]), 0, Tests[i].size, 0, NULL);
		TEST_EVAL( size == Tests[i].size ); 
		
		fprintf(stderr, "Testing fetched data integrity ...");
		res = test_data_integrity( &(Buffer[0]), Tests[i].data, size, test_flags );
		TEST_EVAL( res == 0 ); 
	}	 

	fprintf( stderr, "%d :memory used %d #####################################################\n", __LINE__, UsedMemory );
	SHOW_TIME("Pass 5", started);
	if( interactive )
	   fgetc(stdin);
	for( i = 0 ; i < all_test_count ; ++i ) 
	{
		int size ;
		int r = random();
		if( (r&0x01) == 0 || Tests[i].id == 0 ) 
			continue;
		fprintf(stderr, "%d: Testing forget_data for id %lX size = %d ...\n", __LINE__, Tests[i].id, Tests[i].size);
		forget_data(storage, Tests[i].id);
		size = fetch_data(storage, Tests[i].id, &(Buffer[0]), 0, Tests[i].size, 0, NULL);
		TEST_EVAL( size != Tests[i].size ); 
		Tests[i].id = 0;
#ifndef DEBUG_ALLOCS
		free( Tests[i].data );
#else
		guarded_free( Tests[i].data );
#endif
		Tests[i].data = NULL ; 
		Tests[i].size = 0 ;
	}	 

	fprintf( stderr, "%d :memory used %d #####################################################\n", __LINE__, UsedMemory );
	SHOW_TIME("Pass 6", started);
	if( interactive )
		fgetc(stdin);
	for( k = 0 ; k < 50 ; ++k )
	{
		fprintf( stderr, "%d :dup_data test iteration #%d !!!!!!!!!!!!!!!\n", __LINE__, k );
		for( i = 0 ; i < all_test_count ; ++i ) 
		{
			int k, size, res ;
		
			if( Tests[i].id != 0 ) 
				continue;
			
			for( k = i+1 ; k < all_test_count ; ++k ) 
				if( Tests[k].id != 0 ) 
					break;
			if( k >= all_test_count ) 
				for( k = i ; k >= 0 ; --k ) 
					if( Tests[k].id != 0 ) 
						break;

			if( Tests[k].id == 0 ) 
				continue;
	
			fprintf(stderr, "Testing dup_data for id %lX size = %d ...\n", Tests[k].id, Tests[k].size);
			Tests[i].id = dup_data(storage, Tests[k].id );
			TEST_EVAL( Tests[i].id != 0 ); 
			fprintf(stderr, "Dupped to id %lX - Testing dupped data fetching ...\n", Tests[i].id);
			Tests[i].size = Tests[k].size ;
			Tests[i].data = Tests[k].data ;
			Tests[i].linked = True ;
			size = fetch_data(storage, Tests[i].id, &(Buffer[0]), 0, Tests[i].size, 0, NULL);
			TEST_EVAL( size == Tests[i].size ); 
		
			fprintf(stderr, "Testing dupped data integrity ...\n");
			res = test_data_integrity( &(Buffer[0]), Tests[i].data, size, test_flags );
			TEST_EVAL( res == 0 ); 
	
		}	 

		fprintf( stderr, "%d :memory used %d #####################################################\n", __LINE__, UsedMemory );
		if( interactive )
	   	fgetc(stdin);
		for( i = 0 ; i < all_test_count ; ++i ) 
		{
			int size ;
			int r = random();
			if( (r&0x01) == 0 || Tests[i].id == 0 ) 
				continue;
			fprintf(stderr, "%d: Testing forget_data for id %lX size = %d ...\n", __LINE__, Tests[i].id, Tests[i].size);
			forget_data(storage, Tests[i].id);
			size = fetch_data(storage, Tests[i].id, &(Buffer[0]), 0, Tests[i].size, 0, NULL);
			TEST_EVAL( size != Tests[i].size ); 
			Tests[i].id = 0;
			if( !Tests[i].linked ) 
			{	
				int z ;
				for( z = 0 ; z < all_test_count ; ++z ) 
				{
					if( Tests[z].linked ) 
						if( Tests[z].data == Tests[i].data ) 
						{
							Tests[z].linked = False ;
							Tests[i].data = NULL ; 
							break;
						}
				}	 
				if( Tests[i].data )
				{	
	#ifndef DEBUG_ALLOCS
					free( Tests[i].data );
	#else
					guarded_free( Tests[i].data );
	#endif
				}
			}else
				Tests[i].linked = False ;
			Tests[i].data = NULL ; 
			Tests[i].size = 0 ;
		}	 
	}	
	fprintf( stderr, "%d :memory used %d #####################################################\n", __LINE__, UsedMemory );
	SHOW_TIME("Pass 7", started);
	if( interactive )
	   fgetc(stdin);
	kind = 0 ; 
	min_size = 1 ;
	max_size = StorageTestKinds[kind][0] ; 
	test_count = StorageTestKinds[kind][1] ;
	for( i = 0 ; i < all_test_count ; ++i ) 
	{
		if( Tests[i].id == 0 ) 
		{	
			make_storage_test_data( &(Tests[i]), min_size, max_size, test_flags );
			fprintf(stderr, "Testing store_data for data %p size = %d, and flags 0x%lX...\n", Tests[i].data, Tests[i].size,
					test_flags);
			Tests[i].id = store_data( storage, Tests[i].data, Tests[i].size, test_flags, 0 );
			TEST_EVAL( Tests[i].id != 0 ); 
		}
		if( --test_count <= 0 )
		{
			if( ++kind >= all_test_count ) 
				break;
			min_size = max_size ;
			max_size = StorageTestKinds[kind][0] ; 
			test_count = StorageTestKinds[kind][1] ;
		}		   
	}	 
	fprintf( stderr, "%d :memory used %d #####################################################\n", __LINE__, UsedMemory );
	fprintf( stderr, "%d :compressed_size = %d, uncompressed_size = %d, ratio = %d %% ###########\n", __LINE__, CompressedSize, UncompressedSize, (UncompressedSize<100)?0:(CompressedSize/(UncompressedSize/100)) );
	SHOW_TIME("", started);
	fprintf(stderr, "Testing storage destruction ...");
	destroy_asstorage(&storage);
	TEST_EVAL( storage == NULL ); 

	for( i = 0 ; i < all_test_count ; ++i ) 
		if( Tests[i].data ) 
		{
			if( !Tests[i].linked ) 
			{	
								int z ;
				for( z = 0 ; z < all_test_count ; ++z ) 
				{
					if( Tests[z].linked ) 
						if( Tests[z].data == Tests[i].data ) 
						{
							Tests[z].linked = False ;
							Tests[i].data = NULL ; 
							break;
						}
				}	 
				if( Tests[i].data ) 
				{	
#ifndef DEBUG_ALLOCS
					free( Tests[i].data );
#else
					guarded_free( Tests[i].data );
#endif
				}
			}
			Tests[i].data = NULL ;
			Tests[i].size = 0 ;
		}

	return 0 ;
}

int main(int argc, char **argv )
{
	Bool interactive = False ; 
	ASImage *im = NULL ;
	int i ;
	int res = 0;
	int	test_count = STORAGE_TEST_COUNT ;
	
	set_output_threshold( 10 );
	
	for( i = 1 ; i < argc ; ++i ) 
	{	
		fprintf( stderr, "i = %d argv = \"%s\"\n", i, argv[i] );
		if( strcmp(argv[i], "-i") == 0 ) 
			interactive = True ; 
		else if( i+1 <= argc && strcmp(argv[i], "-s") == 0 ) 
		{
			fprintf( stderr, "Loading test source image \"%s\"\n", argv[i+1] );
			im = file2ASImage( argv[i+1], 0xFFFFFFFF, SCREEN_GAMMA, 0, NULL );	  
			++i ;
		}else if( i+1 <= argc && strcmp(argv[i], "-c") == 0 ) 
		{
			test_count = atoi( argv[i+1] );
			if( test_count > STORAGE_TEST_COUNT ) 
				test_count = STORAGE_TEST_COUNT ;

			fprintf( stderr, "Test count = %d(\"%s\")\n", test_count, argv[i+1] );
			++i ;
		}else if( i+1 <= argc && strcmp(argv[i], "-l") == 0 ) 
		{
			if( freopen( argv[i+1], "w", stderr ) == NULL )
				fprintf( stderr, "Failed to open log file \"%s\"\n", argv[i+1] );
			++i ;
		}

	}

	if( im ) 
	{	
		imdec = start_image_decoding(NULL, im, SCL_DO_ALL, 0, 0, im->width, im->height, NULL);
		fprintf( stderr, "imdec = %p\n", imdec );
	}
	fprintf(stderr, "running tests ( res = %d ) ...\n", res );	
	if( res == 0 )
		res = test_asstorage(interactive, test_count, 0);
#if 1
	if( res == 0 )
		res = test_asstorage(interactive, test_count, ASStorage_RLEDiffCompress);
	if( res == 0 )
		res = test_asstorage(interactive, test_count, ASStorage_RLEDiffCompress|ASStorage_Bitmap);
	if( res == 0 )
		res = test_asstorage(interactive, test_count, ASStorage_32Bit);
	if( res == 0 )
		res = test_asstorage(interactive, test_count, ASStorage_32Bit|ASStorage_RLEDiffCompress);
	if( res == 0 )
		res = test_asstorage(interactive, test_count, ASStorage_32Bit|ASStorage_RLEDiffCompress|ASStorage_Bitmap);
	
	if( res == 0 )
		res = test_asstorage(interactive, test_count, ASStorage_32Bit|ASStorage_8BitShift);
	if( res == 0 )
		res = test_asstorage(interactive, test_count, ASStorage_32Bit|ASStorage_8BitShift|ASStorage_RLEDiffCompress);
	if( res == 0 )
		res = test_asstorage(interactive, test_count, ASStorage_32Bit|ASStorage_8BitShift|ASStorage_RLEDiffCompress|ASStorage_Bitmap);
#endif	
	stop_image_decoding( &imdec );
	return res;
}
#endif

