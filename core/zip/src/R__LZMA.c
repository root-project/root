#include "R__LZMA.h"
#include "lzma.h"
#include <stdio.h>

void R__zipLZMA(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{
   *irep = 0;

   if (*tgtsize <= 0) {
      return;
   }

   if (*srcsize > 0xffffff || *srcsize < 0) {
      return;
   }

   lzma_stream stream = LZMA_STREAM_INIT;
   lzma_ret returnStatus;
   if (cxlevel > 9) cxlevel = 9;
   returnStatus = lzma_easy_encoder(&stream,
                                    (uint32_t)(cxlevel),
                                    LZMA_CHECK_CRC32);
   if (returnStatus != LZMA_OK) {
      return;
   }

   stream.next_in   = (const uint8_t *)src;
   stream.avail_in  = (size_t)(*srcsize);

   stream.next_out  = (uint8_t *)(&tgt[kHeaderSize]);
   stream.avail_out = (size_t)(*tgtsize);

   returnStatus = lzma_code(&stream, LZMA_FINISH);
   if (returnStatus != LZMA_STREAM_END) {
      /* No need to print an error message. We simply abandon the compression
         the buffer cannot be compressed or compressed buffer would be larger than original buffer
      */
      lzma_end(&stream);
      return;
   }
   lzma_end(&stream);


   tgt[0] = 'X';  /* Signature of LZMA from XZ Utils */
   tgt[1] = 'Z';
   tgt[2] = 0;

   unsigned in_size   = (unsigned) (*srcsize);
   uint64_t out_size  = stream.total_out;             /* compressed size */

   tgt[3] = (char)(out_size & 0xff);
   tgt[4] = (char)((out_size >> 8) & 0xff);
   tgt[5] = (char)((out_size >> 16) & 0xff);

   tgt[6] = (char)(in_size & 0xff);         /* decompressed size */
   tgt[7] = (char)((in_size >> 8) & 0xff);
   tgt[8] = (char)((in_size >> 16) & 0xff);

   *irep = stream.total_out + kHeaderSize;
}

void R__unzipLZMA(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep)
{
   *irep = 0;

   lzma_stream stream = LZMA_STREAM_INIT;
   lzma_ret returnStatus;

   returnStatus = lzma_stream_decoder(&stream, 
                                      UINT64_MAX,
                                      0U);
   if (returnStatus != LZMA_OK) {
      fprintf(stderr,
              "R__unzipLZMA: error %d in lzma_stream_decoder\n",
              returnStatus);
      return;
   }

   stream.next_in   = (const uint8_t *)(&src[kHeaderSize]);
   stream.avail_in  = (size_t)(*srcsize);
   stream.next_out  = (uint8_t *)tgt;
   stream.avail_out = (size_t)(*tgtsize);

   returnStatus = lzma_code(&stream, LZMA_FINISH);
   if (returnStatus != LZMA_STREAM_END) {
      fprintf(stderr,
              "R__unzipLZMA: error %d in lzma_code\n",
              returnStatus);
      lzma_end(&stream);
      return;
   }
   lzma_end(&stream);

   *irep = stream.total_out;
}
