#ifndef __OOUC_TOKENIZER__
#define __OOUC_TOKENIZER__
/******************************************************************************/
/*                                                                            */
/*                    X r d O u c T o k e n i z e r . h h                     */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//       $Id$

class XrdOucTokenizer
{
public:

            XrdOucTokenizer(char *bp) {Attach(bp);}

           ~XrdOucTokenizer() {}

// Attach a new buffer to the tokenizer.
//
void         Attach(char *bp);

// Get the next record from a buffer. Return null upon eof or error.
//
char        *GetLine();

// Get the next blank-delimited token in the record returned by Getline(). A
// null pointer is returned if no more tokens remain. Each token is terminated
// a null byte. Note that the record buffer is modified during processing. The
// routine may optionally return a pointer to the remainder of the line with 
// no leading blanks. The lowcase argument, if 1, converts all letters to lower 
// case in the token.
//
char        *GetToken(char **rest=0, int lowcase=0);

// RetToken() simply backups the token scanner the last tken returned. Only
// one backup is allowed.
//
void         RetToken();

// A 0 indicates that tabs in the stream should be converted to spaces.
// A 1 inducates that tabs should be left alone (the default).
//
void         Tabs(int x=1) {notabs = !x;}

/******************************************************************************/
  
private:
        char *buff;
        char *token;
        char *tnext;
        int   notabs;
};
#endif
