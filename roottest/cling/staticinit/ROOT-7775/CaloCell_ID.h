#include "ROOT-7775/prelude.h"

#ifndef LAREM_ID_H
#define LAREM_ID_H

#include "ROOT-7775/CLASS_DEF.h"

class AtlasDetectorID // : public IdHelper
{
};

//using the macros below we can assign an identifier (and a version)
////This is required and checked at compile time when you try to record/retrieve
CLASS_DEF(AtlasDetectorID, 164875623, 1)

//#include "t01/AtlasDetectorID.h"
//from #include "CaloIdentifier/LArEM_Base_ID.h"



#include "ROOT-7775/BaseInfo.h"

class LArEM_Base_ID2 {};

class LArEM_ID : public LArEM_Base_ID2
{
public:        

  //typedef Identifier::size_type  size_type ;

  LArEM_ID(void);    
  ~LArEM_ID(void);


  
  /** initialization from the identifier dictionary*/
  //virtual int  initialize_from_dictionary (const IdDictMgr& dict_mgr);
};

//using the macro below we can assign an identifier (and a version)
//This is required and checked at compile time when you try to record/retrieve
// CLASS_DEF( LArEM_ID , 163583365 , 1 )
SG_BASE (LArEM_ID, LArEM_Base_ID2);


#endif // LAREM_ID_H
