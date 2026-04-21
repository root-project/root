/* @(#)root/x11:$Id$ */
/* Author: S.Linev   20/04/2026 */
/* C++ interface for gifdecode.c */

#ifndef gifdecode_h
#define gifdecode_h

class TGifDecode {
   private:
      unsigned char  *ptr1 = nullptr;     /* pointer to GIF array */
      unsigned char  *ptr2 = nullptr;     /* pointer to PIX array */

      int      CurCodeSize = 0;           /* current number of bits per code */
      int      CurMaxCode = 0;            /* maximum code, given CurCodeSize */
      long     CurBit = 0;                /* current bit in GIF image data */

      int ReadCode();
      void OutPixel(unsigned char pix);

   public:

      static int GIFinfo(unsigned char *GIFarr, int *Width, int *Height, int *Ncols);

      int GIFdecode(unsigned char *GIFarr, unsigned char *PIXarr, int *Width, int *Height, int *Ncols, unsigned char *R, unsigned char *G, unsigned char *B);

};


#endif