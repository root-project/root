// @(#)root/tree:$Name:  $:$Id: TChainElement.h,v 1.2 2000/11/21 20:47:29 brun Exp $
// Author: Rene Brun   11/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TChainElement
#define ROOT_TChainElement


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TChainElement                                                        //
//                                                                      //
// Describes a component of a TChain.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TChainElement : public TNamed {

protected:
    Int_t         fEntries;         //Number of entries in the tree of this chain element
    Int_t         fNPackets;        //Number of packets
    Int_t         fPacketSize;      //Number of events in one packet for parallel root
    Int_t         fStatus;          //branch status when used as a branch
    void         *fBaddress;        //!branch address when used as a branch
    char         *fPackets;         //!Packet descriptor string

public:
        TChainElement();
        TChainElement(const char *filename, const char *title);
        virtual ~TChainElement();
        virtual void     CreatePackets();
        virtual void    *GetBaddress() const {return fBaddress;}
        virtual char    *GetPackets() const {return fPackets;}
        virtual Int_t    GetPacketSize() const {return fPacketSize;}
        virtual Int_t    GetStatus() const {return fStatus;}
        virtual void     SetBaddress(void *add) {fBaddress = add;}
        virtual void     SetNumberEntries(Int_t n) {fEntries=n;}
        virtual void     SetPacketSize(Int_t size = 100);
        virtual void     SetStatus(Int_t status) {fStatus = status;}

        ClassDef(TChainElement,0)  //A chain element
};

#endif

