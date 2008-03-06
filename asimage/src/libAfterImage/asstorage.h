#ifndef ASSTORAGE_H_HEADER_INCLUDED
#define ASSTORAGE_H_HEADER_INCLUDED


#define AS_STORAGE_PAGE_SIZE		4096

/* 
 *	there could be up to 16 arrays of 1024 pointers to slots each in Storage Block
 *	There could be 2^18 StorageBlocks in ASStorage
 */
#define AS_STORAGE_SLOTS_BATCH		1024  /* we allocate pointers to slots in batches of one page eache  */ 
#define AS_STORAGE_SLOT_ID_BITS		14  /* 32*512 == 2^14 */ 
#define AS_STORAGE_MAX_SLOTS_CNT	(0x01<<AS_STORAGE_SLOT_ID_BITS)

#define AS_STORAGE_BLOCK_ID_BITS	(32-AS_STORAGE_SLOT_ID_BITS)
#define AS_STORAGE_MAX_BLOCK_CNT   	(0x01<<AS_STORAGE_BLOCK_ID_BITS)
/* #define AS_STORAGE_DEF_BLOCK_SIZE	(1024*256)  */
#define AS_STORAGE_DEF_BLOCK_SIZE	(1024*128)  /* 128 Kb */  
#define AS_STORAGE_NOUSE_THRESHOLD	(1024*8)  /* 8 Kb if total_free < 8K we should not try and use that 
											   *  block as we may fall into trap constantly defragmenting it
											   *  so we prefer to leave memory unused since 2 pages is not too much to loose */  


#define ASStorageSlot_SIZE 16 /* 16 bytes */
#define ASStorageSlot_USABLE_SIZE(slot) (((slot)->size+15)&0x8FFFFFF0)
#define ASStorageSlot_FULL_SIZE(slot) (ASStorageSlot_USABLE_SIZE(slot)+ASStorageSlot_SIZE)
/* space for slots is allocated in 16 byte increments */
#define AS_STORAGE_GetNextSlot(slot) ((slot)+1+(ASStorageSlot_USABLE_SIZE(slot)>>4))


/* RLE encoding of difference 
 * We calculate difference between following bytes. If differece is zero - its RLE encoded.
 * If its +-1 - its encoded as 2 bit values
 * If Its +-(from 2 to 7) - its encoded using 4 bit values
 * If Its +-(from 8 to 127) - its encoded using 8 bit values  
 * If Its +-(from 128 to 255) - its encoded using 9 bit values
 * 
 * The hope is that most of the bytes will be reduced to 0 
 * The next likely value will be from 2 to 7 
 * and only few cases will fall in other categories
 * 
 * For bitmaps we store lengths of ones and zerous, assuming that each string tsarts with 0
 * 
 * */

/* The following lines is used only for non-bitmaps : */
#define RLE_ZERO_MASK				0x0080  /* M        */  
#define RLE_ZERO_LENGTH				0x007F  /*  LLLLLLL */  
#define RLE_ZERO_SIG				0x0000  /* 0LLLLLLL - identical to a string of LLLLLLL zeros */  

#define RLE_NOZERO_SHORT_MASK		0x00C0  /* MM       */  
#define RLE_NOZERO_SHORT_LENGTH		0x003F  /*   LLLLLL */  
#define RLE_NOZERO_SHORT_SIG		0x00C0  /* 11LLLLLL followed by stream of LLLLLL 4 or 2 bit values */

#define RLE_NOZERO_LONG_MASK		0x00F0  /* MMMM     */  
#define RLE_NOZERO_LONG_LENGTH		0x000F  /*     LLLL */  
#define RLE_NOZERO_LONG1_SIG		0x00A0  /* 1010LLLL followed by stream of LLLL 2 or 4 bit values */  
#define RLE_NOZERO_LONG2_SIG		0x00B0  /* 1011LLLL followed by stream of LLLL 1 byte values */  

#define RLE_9BIT_SIG				0x0080  /* 1000LLLL followed by stream of LLLL 1 byte values 
                                               that change sign from byte to byte starting with positive */     
#define RLE_9BIT_NEG_SIG	  		0x0090  /* 1001LLLL followed by stream of LLLL 1 byte values 
                                               that change sign from byte to byte starting with negative */     

#define AS_STORAGE_DEFAULT_BMAP_THRESHOLD 0x7F
#define AS_STORAGE_DEFAULT_BMAP_VALUE	  0xFF


typedef struct ASStorageSlot
{
/* Pointer to ASStorageSlot is the pointer to used memory beginning - ASStorageSlot_SIZE 
 * thus we need not to store it separately 
 */
#define ASStorage_ZlibCompress		(0x01<<0)  /* do we really want that ? */ 
#define ASStorage_RLEDiffCompress 	(0x01<<1)  /* RLE of difference */ 

#define ASStorage_CompressionType	(0x0F<<0)  /* allow for 16 compression schemes */
#define ASStorage_Used				(0x01<<4)
#define ASStorage_NotTileable		(0x01<<5)
#define ASStorage_Reference			(0x01<<6)  /* data is the id of some other slot */ 
#define ASStorage_Bitmap			(0x01<<7)  /* data is 1 bpp */ 
#define ASStorage_32Bit				(0x01<<8)  /* data is 32 bpp with only first 8 bits being significant */ 
#define ASStorage_BitShiftFlagPos   9
#define ASStorage_BitShift			(0x03<<ASStorage_BitShiftFlagPos)  
#define ASStorage_8BitShift			(0x01<<ASStorage_BitShiftFlagPos)  
												/* data is 32 bpp shifted left by 8 bit 
												 * (must combine with _32Bit flag )*/ 
#define ASStorage_16BitShift		(0x01<<(ASStorage_BitShiftFlagPos+1)) 
												/* data is 32 bpp shifted left by 16 bit 
												* (must combine with _32Bit flag )
												* If combined with 8BitShift - results in 24 bit shift */ 
#define ASStorage_24BitShift		(ASStorage_8BitShift|ASStorage_16BitShift)
#define ASStorage_Flags2ShiftIdx(f) (((f)>>ASStorage_BitShiftFlagPos)&0x03)
#define ASStorage_Flags2Shift(f) 	(ASStorage_Flags2ShiftIdx(f)*8)
#define ASStorage_Masked			(0x01<<11) /* mask 32bit value to filter out higher 24 bits
                                                * if combined with BitShift - bitshift is done 
												* prior to masking */ 


#define ASStorage_32BitRLE			(ASStorage_RLEDiffCompress|ASStorage_32Bit)

	CARD16  flags ;
	CARD16  ref_count ;
	CARD32  size ;
	CARD32  uncompressed_size ;
	CARD16  index ;  /* reverse mapping of slot address into index in array */
	/* slots may be placed in array pointing into different areas of the memory 
	 * block, since we will need to implement some sort of garbadge collection and 
	 * defragmentation mechanism - we need to be able to process them in orderly 	
	 * fashion. 
	 * So finally : 
	 * 1) slot's index does not specify where in the memory slot 
	 * is located, it is only used to address slot from outside.
	 * 2) Using slots memory address and its size we can go through the chain of slots
	 * and perform all the maintenance tasks  as long as we have reverse mapping 
	 * of addresses into indexes.
	 * 
	 */
	CARD16 reserved ;          /* to make us have size rounded by 16 bytes margin */
	/* Data immidiately follows here : 
	 * CARD8   data[0] ; */

#define ASStorage_Data(s)  ((CARD8*)((s)+1))

}ASStorageSlot;

/* turns out there is no performance gains from using int here instead of short - 
so save some memory if we can : */
typedef short ASStorageDiff;


typedef void (*compute_diff_func_type)(ASStorageDiff*,CARD8*,int);
typedef int  (*copy_data32_func_type)(CARD8*,CARD32*,int);
typedef int  (*copy_data32_tinted_func_type)(CARD8*,CARD32*,int,CARD32);


typedef struct ASStorageBlock
{
#define ASStorage_MonoliticBlock		(0x01<<0) /* block consists of a single batch of storage */
 	CARD32  flags ;
	int 	size ;

	int   	total_free;
	ASStorageSlot  *start, *end;
	/* array of pointers to slots is allocated separately, so that we can reallocate it 
	   in case we have lots of small slots */
	ASStorageSlot **slots;
	int slots_count, unused_count ;
	int first_free, last_used ;
	int long_searches ;

}ASStorageBlock;

typedef struct ASStorage
{
	int default_block_size ;


	ASStorageBlock **blocks ;
	int 			blocks_count;

	ASStorageDiff  *diff_buf ;
	CARD8  *comp_buf ;
	size_t 	comp_buf_size ; 

}ASStorage;


typedef CARD32 ASStorageID ;

ASStorageID store_data(ASStorage *storage, CARD8 *data, int size, ASFlagType flags, CARD8 bitmap_threshold);
ASStorageID store_data_tinted(ASStorage *storage, CARD8 *data, int size, ASFlagType flags, CARD16 tint);

/* data will be fetched fromthe slot identified by id and placed into buffer. 
 * Data will be fetched from offset  and will count buf_size bytes if buf_size is greater then
 * available data - data will be tiled to accomodate this size, unless NotTileable is set */
int  fetch_data(ASStorage *storage, ASStorageID id, CARD8 *buffer, int offset, int buf_size, CARD8 bitmap_value, int *original_size);
int  fetch_data32(ASStorage *storage, ASStorageID id, CARD32 *buffer, int offset, int buf_size, CARD8 bitmap_value, int *original_size);
int  threshold_stored_data(ASStorage *storage, ASStorageID id, unsigned int *runs, int width, unsigned int threshold);

/* slot identified by id will be marked as unused */
void forget_data(ASStorage *storage, ASStorageID id);

void print_storage(ASStorage *storage);

int print_storage_slot(ASStorage *storage, ASStorageID id);
Bool query_storage_slot(ASStorage *storage, ASStorageID id, ASStorageSlot *dst );

/* returns new ID without copying data. Data will be stored as copy-on-right. 
 * Reference count of the data will be increased. If optional dst_id is specified - 
 * its data will be erased, and it will point to the data of src_id: 
 */				
ASStorageID dup_data(ASStorage *storage, ASStorageID src_id);

/* this will provide access to default storage heap that is used whenever above functions get
 * NULL passed as ASStorage parameter :
 */
void flush_default_asstorage();
int set_asstorage_block_size( ASStorage *storage, int new_size );


#endif
