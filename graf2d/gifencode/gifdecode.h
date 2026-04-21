/* @(#)root/x11:$Id$ */
/* Author: S.Linev   20/04/2026 */
/* C++ interface for gifdecode.c */

#ifndef gifdecode_h
#define gifdecode_h

class TGifDecode {
   private:
      unsigned char  *ptr1 = nullptr;     /* pointer to GIF array */
      unsigned char  *ptr2 = nullptr;     /* pointer to PIX array */

      int           CurCodeSize = 0;           /* current number of bits per code */
      int           CurMaxCode = 0;            /* maximum code, given CurCodeSize */
      long          CurBit = -1;                /* current bit in GIF image data */

      long          b3[3] = {0, 0, 0};
      long          CurByte = -1;
      unsigned char lblk = 0;

      int ReadCode();
      void OutPixel(unsigned char pix);

   public:

      static int GIFinfo(unsigned char *GIFarr, int *Width, int *Height, int *Ncols);

      int GIFdecode(unsigned char *GIFarr, unsigned char *PIXarr, int *Width, int *Height, int *Ncols, unsigned char *R, unsigned char *G, unsigned char *B);

};


#endif