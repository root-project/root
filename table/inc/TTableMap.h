#ifndef ROOT_ATTABLEMAP_T
#define ROOT_ATTABLEMAP_T

#include "assert.h"
#include <vector>
#include "TTable.h"

//////////////////////////////////////////////////////
//
// Class TTableMap
// Iterator of the table with extra index array
//
//////////////////////////////////////////////////////


class TTableMap : public TObject
#ifndef __CINT__
 , public std::vector<Long_t>  
#endif
{
  protected:
     TTable  *fTable;         // pointer to the refered TTable

  public:
	
    TTableMap(const TTable *table=0);
    TTableMap(const TTableMap &map) : TObject(map)
#ifndef __CINT__
		, std::vector<Long_t>(map)
#endif
    , fTable(map.fTable)	{;}
    ~TTableMap(){;}
    Bool_t  IsValid() const;
    Bool_t  IsFolder() const;
    void Push_back(Long_t next); // workaround for Cint
    TTable *Table(){return fTable;}

    TTable::iterator Begin();
    TTable::iterator Begin() const;
    TTable::iterator End();
    TTable::iterator End()   const;
    
    ClassDef(TTableMap,1) // "Map" array for TTable object
};

//___________________________________________________________________________________________________________
inline  Bool_t TTableMap::IsValid() const
{
    // Check whether all "map" values do belong the table
    TTable::iterator i      = Begin();
    TTable::iterator finish = End();
    Int_t totalSize          = fTable->GetNRows();

    for (; i != finish; i++) {
        Long_t th = *i;
        if (  th == -1 || (0 <= th && th < totalSize) ) continue;
        return kFALSE;
    }
    return kTRUE;
}
//___________________________________________________________________________________________________________
inline TTable::iterator TTableMap::Begin()          { return TTable::iterator(*fTable, this->begin());}
//___________________________________________________________________________________________________________
inline TTable::iterator TTableMap::Begin()    const { return TTable::iterator(*fTable, ((TTableMap*) this)->begin());}
//___________________________________________________________________________________________________________
inline TTable::iterator TTableMap::End()            { return TTable::iterator(*fTable, this->end());}
//___________________________________________________________________________________________________________
inline TTable::iterator TTableMap::End()      const { return TTable::iterator(*fTable, ((TTableMap*) this)->end());}
//___________________________________________________________________________________________________________
inline Bool_t           TTableMap::IsFolder() const { return kTRUE;}
//___________________________________________________________________________________________________________
inline void             TTableMap::Push_back(Long_t next){ push_back(next); } // workaround for Cint


#endif
