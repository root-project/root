\page freesegments Format of FreeSegments record

Format of FreeSegments record, release 6.22.06.  It is never compressed.
It is probably not accessed by its key, but from its offset given in the file header.

If any *individual* free segments refer to bytes beyond 2000000000,
their fFirst/fLast have 8 bytes, not 4 and 1000 is added to the TFree Version.

Some free segment records may be 32 bit while others are 64 bit.

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
 ----------TKey---------------
  byte 0->3           Nbytes    = Number of bytes in compressed record (TKey+data)         TKey::fNbytes
       4->5           Version   = TKey class version identifier                            TKey::fVersion
       6->9           ObjLen    = Number of bytes of uncompressed data                     TKey::fObjLen
      10->13          Datime    = Date and time when record was written to file            TKey::fDatime
                                | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
      14->15          KeyLen    = Number of bytes in key structure  (TKey)                 TKey::fKeyLen
      16->17          Cycle     = Cycle of key                                             TKey::fCycle
      18->21 [18->25] SeekKey   = Byte offset of record itself (consistency check)         TKey::fSeekKey
      22->25 [26->33] SeekPdir  = Byte offset of parent directory record (TFile)           TKey::fSeekPdir
      26->26 [34->34] lname     = Number of bytes in the class name (5)                    TKey::fClassName
      27->.. [35->..] ClassName = Object Class Name ("TFile")                              TKey::fClassName
       0->0           lname     = Number of bytes in the object name                       TNamed::fName
       1->..          Name      = lName bytes with the name of the object `<file-name>`    TNamed::fName
       0->0           lTitle    = Number of bytes in the object title                      TNamed::fTitle
       1->..          Title     = lTitle bytes with the title of the object `<file-title>` TNamed::fTitle
 ----------DATA---------------
       0->1           Version   = TFree class version identifier                           TFree::Class_Version()
       2->5  [ 2-> 9] fFirst    = First free byte of first free segment                    TFree::fFirst
       6->9  [10->17] fLast     = Last free byte of first free segment (inclusive)         TFree::fLast
                                  (e.g. a free segment that is 1 byte long would have fFirst == fLast)
       ....           Sequentially, Version, fFirst and fLast of additional free segments.
       ....           There is always one free segment beginning at file end and ending before 2000000000.
       ....           If the file size is larger than 2000000000, the last segment ends with 4000000000.
</pre></div>
