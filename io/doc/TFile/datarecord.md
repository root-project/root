\page datarecord Format of data records

### Release 3.02.06
  A ROOT file is mostly a suite of consecutive data records with the following format
 <Name>;<Cycle> uniquely identifies the record in a directory

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
 ----------TKey-(never compressed)--------------
  byte 0->3  Nbytes    = Number of bytes in compressed record (Tkey+data)   TKey::fNbytes
       4->5  Version   = TKey class version identifier                      TKey::fVersion
       6->9  ObjLen    = Number of bytes of uncompressed data               TKey::fObjLen
      10->13 Datime    = Date and time when record was written to file      TKey::fDatime
                       | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
      14->15 KeyLen    = Number of bytes in key structure (TKey)            TKey::fKeyLen
      16->17 Cycle     = Cycle of key (e.g. 1)                              TKey::fCycle
      18->21 SeekKey   = Byte offset of record itself (consistency check)   TKey::fSeekKey
      22->25 SeekPdir  = Byte offset of parent directory record             TKey::fSeekPdir
      26->26 lname     = Number of bytes in the class name                  TKey::fClassName
      27->.. ClassName = Object Class Name                                  TKey::fClassName
       0->0  lname     = Number of bytes in the object name                 TNamed::fName
       1->.. Name      = lName bytes with the name of the object            TNamed::fName
       0->0  lTitle    = Number of bytes in the object title                TNamed::fTitle
       1->.. Title     = lTitle bytes with the title of the object          TNamed::fTitle
 ----------DATA---(may be compressed)-----------
       0->..             The data object itself.  For an example, see \ref dobject
</pre></div>
