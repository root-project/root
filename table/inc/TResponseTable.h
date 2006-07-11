// @(#)root/table:$Name:  $Id: TResponseTable.h,v 1.1 2003/01/27 20:41:36 brun Exp $
// Author: Valery Fine(fine@bnl.gov)   30/06/2001

#include "TGenericTable.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TResponseTable                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TResponseTable : public TGenericTable
{
    public:
      TResponseTable();
      TResponseTable(const char *name,const char *volumepath, const char *responseDefintion, Int_t allocSize);
      virtual ~TResponseTable(){}
      void SetResponse(int track, int *nvl, float *response);
      static Int_t FindResponseLocation(TTableDescriptor  &dsc);

    protected: 
      void AddVolumePath(const char *path);
      void AddResponse(const char *chit);
      void AddElement(const char *path,EColumnType type);
    private:
      Int_t fResponseLocation;

     ClassDef(TResponseTable,4) // Generic Geant detector response table
};
