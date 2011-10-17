// @(#)root/net:$Id$
// Author: Philippe Canal October 2011.

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TParallelMergingFile
#define ROOT_TParallelMergingFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TParallelMergingFile                                                 //
//                                                                      //
// Specialization of TMemFile to connect to a parallel file merger.     //
// Upon a call to UploadAndReset, the content already written to the    //
// file is upload to the server and the object implementing the function//
// ResetAfterMerge (like TTree) are reset.                              //
// The parallel file merger will then collate the information coming    //
// from this client and any other client in to the file described by    //
// the filename of this object.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMemFile
#include "TMemFile.h"
#endif
#ifndef ROOT_TMessage
#include "TMessage.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif


class TSocket;
class TArrayC;

class TParallelMergingFile : public TMemFile 
{
private:
   TSocket *fSocket;         // Socket to the parallel file merger server.
   TUrl     fServerLocation; // Url of the server.
   Int_t    fServerIdx;      // Index of this socket/file on the server.
   Int_t    fServerVersion;  // Protocol version used by the server.
   TArrayC *fClassSent;      // Record which StreamerInfo we already sent.
   TMessage fMessage;

public:
   TParallelMergingFile(const char *filename, Option_t *option = "", const char *ftitle = "", Int_t compress = 1);   
   ~TParallelMergingFile();

   virtual void   Close(Option_t *option="");
           Bool_t UploadAndReset();
   virtual Int_t  Write(const char *name=0, Int_t opt=0, Int_t bufsiz=0);
   virtual Int_t  Write(const char *name=0, Int_t opt=0, Int_t bufsiz=0) const;
   virtual void   WriteStreamerInfo();

   ClassDef(TParallelMergingFile,2);  // TFile specialization that will semi-automatically upload its content to a merging server.
};

#endif // ROOT_TParallelMergingFile