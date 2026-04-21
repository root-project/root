/* @(#)root/x11:$Id$ */
/* Author: S.Linev   20/04/2026 */
/* C++ interface for gifencode.c */


#ifndef gifencode_h
#define gifencode_h

#include <cstdio>
#include <cstdlib>
#include <cstring>


class TGifEncode {

   private:

      enum { HSIZE = 5003 };               /* hash table size */

      unsigned long cur_accum = 0;
      int           cur_bits = 0;
      int           a_count = 0;

      int      BitsPixel = 0;              /* number of bits per pixel */
      int      IniCodeSize = 0;            /* initial number of bits per code */
      int      CurCodeSize = 0;            /* current number of bits per code */
      int      CurMaxCode = 0;             /* maximum code, given CurCodeSize */
      int      ClearCode = 0;              /* reset code */
      int      EOFCode = 0;                /* end of file code */
      int      FreeCode = 0;               /* first unused entry */

      long     HashTab [HSIZE];           /* hash table */
      int      CodeTab [HSIZE];           /* code table */
      unsigned char  accum[256];

      long fNbyte = 0;
      FILE *fOut = nullptr;

      void put_byte(unsigned char b);
      void     char_init();
      void     char_out(unsigned char c);
      void     char_flush();
      void     put_short(int word);
      void     output(int code);

    protected:

       virtual void get_scline(int y, int width, unsigned char *buf) = 0;

    public:
       virtual ~TGifEncode() { CloseFile(); }

       bool OpenFile(const char *fname, const char *opt = "w+");
       void CloseFile();

       long GIFencode(int Width, int Height, int Ncol, unsigned char *R, unsigned char *G, unsigned char *B);
};


#endif
