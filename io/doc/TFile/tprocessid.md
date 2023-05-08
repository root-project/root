\page tprocessid Format of TProcessID record


Format of TProcessID record in release 3.02.06.
Will be present if there are any referenced objects.

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
 ----------TKey---------------
   byte 0->3  Nbytes   = Number of bytes in compressed record (Tkey+data)   TKey::fNbytes
        4->5  Version  = TKey class version identifier                      TKey::fVersion
        6->9  ObjLen   = Number of bytes of uncompressed data               TKey::fObjLen
       10->13 Datime   = Date and time when record was written to file      TKey::fDatime
                       | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
       14->15 KeyLen   = Number of bytes in key structure (TKey)            TKey::fKeyLen
       16->17 Cycle    = Cycle of key                                       TKey::fCycle
       18->21 SeekKey  = Byte offset of record itself (consistency check)   TKey::fSeekKey
       22->25 SeekPdir = Byte offset of parent directory record             TKey::fSeekPdir
       26->26 lname    = Number of bytes in the class name (10)             TKey::fClassName
       27->36 ClassName= Object Class Name ("TProcessID")                   TKey::fClassName
       37->37 lname    = Number of bytes in the object name                 TNamed::fName
       38->.. Name     = lName bytes with the name of the object            TNamed::fName
                       | (e.g. "ProcessID0")
        0->0  lTitle   = Number of bytes in the object title                TNamed::fTitle
        1->.. Title    = lTitle bytes with the title of the object          TNamed::fTitle
                       | (Identifies processor, time stamp, etc.)
                       | See detailed explanation below.
 ----------DATA--------------
        0->3 ByteCount = Number of remaining bytes in TProcessID object (uncompressed)
                       |   OR'd with kByteCountMask (0x40000000)
        4->5 Version   = Version of TProcessID Class
 -Begin TNamed object (Base class of TProcessID)
        6->9 ByteCount = Number of remaining bytes in TNamed object (uncompressed)
                       |   OR'd with kByteCountMask (0x40000000)
       10->11 Version  = Version of TNamed Class
       12->21          = TObject object (Base class of TNamed) (see \ref tobject).
                       |   The TProcessID object is not itself referenced.
       22->22 lname    = Number of bytes in the object name                 TNamed::fName
       23->.. Name     = lName bytes with the name of the object            TNamed::fName
                       | The name will be "ProcessID" concatenated with
                       | a decimal integer, or "pidf".
        0->0  lTitle   = Number of bytes in the object title                TNamed::fTitle
        1->.. Title    = lTitle bytes with the title of the object          TNamed::fTitle
                       | (Identifies processor, time stamp, etc.)
                       | See detailed explanation below.
 -End TNamed object
</pre></div>

### Explanation of the title of a TProcessID object

The title of a TProcessID object is a globally unique identifier of the
ROOTIO process that created it.  It is derived from the following quantities.

  1. The creation time ("fTime") of the TProcessID record.  This is a 60 bit time
     in 100ns ticks since Oct. 15, 1582.

  2. A 16 bit random unsigned integer ("clockeq") generated from a seed that is the
     job's process ID.  The highest two bits are not used.

  3. A six byte unsigned quantity ("fNode") identifying the machine.  If the machine has a
     valid network address, the first four bytes are set to that address, and the last two bytes
     are stuffed with 0xbe and 0xef respectively.  Otherwise a six byte quantity is generated
     from the time and random machine statistics. In this case, the high order bit of the
     first byte is set to 1, to distinguish it from a network ID, where the bytes can be
     no larger than 255.

We the define the following quantities (class TUUID):
<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
        UInt_t    fTimeLow;               // 60 bit time, lowest 32 bits
        UShort_t  fTimeMid;               // 60 bit time, middle 16 bits
        UShort_t  fTimeHiAndVersion;      // 60 bit time, highest 12 time bits (low 12 bits)
                                          // + 4 UUID version bits (high 4 bits)
                                          // version is 1 if machine has valid network address
                                          // and 3 otherwise.
        UChar_t   fClockSeqHiAndReserved; // high 6 clockseq bits (low 6 bits)
                                          // + 2 high bits reserved (currently set to binary 10)
        UChar_t   fClockSeqLow;           // low 8 clockseq bits
        UChar_t   fNode[6];               // 6 node (machine) id bytes
</pre></div>

Then the following sprintf() call defines the format of the title string:
<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
   sprintf(Title, "%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           fTimeLow, fTimeMid, fTimeHiAndVersion, fClockSeqHiAndReserved,
           fClockSeqLow, fNode[0], fNode[1], fNode[2], fNode[3], fNode[4],
           fNode[5]);
</pre></div>

Since the title written to disk is preceded by its byte count, the delimiting null is not written.
