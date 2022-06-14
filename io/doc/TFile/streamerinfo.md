\page streamerinfo Format of StreamerInfo record

Format of StreamerInfo record in release 3.02.06.
It is probably not accessed by its key, but from its offset given in the file header.
The StreamerInfo record DATA consists of a TList (list) object containing elements
of class TStreamerInfo.

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
 ----------TKey-(never compressed)----------------------
  byte 0->3  Nbytes    = Number of bytes in compressed record (TKey+data)         TKey::fNbytes
       4->5  Version   = TKey class version identifier                            TKey::fVersion
       6->9  ObjLen    = Number of bytes of uncompressed data                     TKey::fObjLen
      10->13 Datime    = Date and time when record was written to file            TKey::fDatime
                       | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
      14->15 KeyLen    = Number of bytes in key structure (TKey) (64)             TKey::fKeyLen
      16->17 Cycle     = Cycle of key                                             TKey::fCycle
      18->21 SeekKey   = Byte offset of record itself (consistency check)         TKey::fSeekKey
      22->25 SeekPdir  = Byte offset of parent directory record (TFile)           TKey::fSeekPdir
      26->26 lname     = Number of bytes in the class name (5)                    TKey::fClassName
      27->31 ClassName = Object Class Name ("TList")                              TKey::fClassName
      32->32 lname     = Number of bytes in the object name (12)                  TNamed::fName
      33->44 Name      = lName bytes with the name of the object ("StreamerInfo") TNamed::fName
      45->45 lTitle    = Number of bytes in the object title (18)                 TNamed::fTitle
      46->63 Title     = lTitle bytes with the title of the object                TNamed::fTitle
                       | ("Doubly linked list")
</pre></div>

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
 ----------TList-(always compressed at level 1 (even if compression level 0))----
</pre></div>
 The DATA is a TList collection object containing TStreamerInfo objects.
 Below is the format of this TList data.

 Here is the format of a TList object in Release 3.02.06.
 Comments and offsets refer specifically to its use in the StreamerInfo record.

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
--------
      0->3  ByteCount = Number of remaining bytes in TList object (uncompressed)
                      |   OR'd with kByteCountMask (0x40000000)
      4->5  Version   = Version of TList Class
      6->15           = TObject object (a base class of TList) (see \ref tobject).
                      |   Objects in StreamerInfo record are not referenced.
                      |   Would be two bytes longer (6->17) if object were referenced.
     16->16 fName     = Number of bytes in name of TList object, followed by the
                      |   name itself.  (TCollection::fName).  The TList object in
                      |   StreamerInfo record is unnamed, so byte contains 0.
     17->20 nObjects  = Number of objects in list.
     21->.. objects   = Sequentially, TStreamerInfo Objects in the list.
                      | In the StreamerInfo record, the objects in the list are
                      |   TStreamerInfo objects.  There will be one TStreamerInfo
                      |   object for every class used in data records other than
                      |   core records and the the StreamerInfo record itself.
-------
</pre></div>

 Here is the format of a TStreamerInfo object in Release 3.02.06.
 Note: Although TStreamerInfo does not use the default streamer, it has the same
 format as if it did.  (compare with \ref dobject)

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
      0->3  ByteCount = Number of remaining bytes in TStreamerInfo object (uncompressed)
                      |   OR'd with kByteCountMask (0x40000000)
      4->.. ClassInfo = Information about TStreamerInfo class
                      | If this is the first occurrence of a TStreamerInfo object in the record
                      |  4->7  -1        = New class tag (constant kNewClassTag = 0xffffffff)
                      |  8->21 Classname = Object Class Name "TStreamerInfo" (null terminated)
                      | Otherwise
                      |  4->7 clIdx      = Byte offset of new class tag in record, plus 2.
                      | OR'd with kClassMask (0x80000000)
      0->3  ByteCount = Number of remaining bytes in TStreamerInfo object (uncompressed)
                      |   OR'd with kByteCountMask (0x40000000)
      4->5  Version   = Version of TStreamerInfo Class
 -Begin TNamed object (Base class of TStreamerInfo)
      6->9  ByteCount = Number of remaining bytes in TNamed object
                      |   OR'd with kByteCountMask (0x40000000)
     10->11 Version   = Version of TNamed Class
     12->21           = TObject object (Base class of TNamed) (see \ref tobject).
                      |   Objects in StreamerInfo record are not referenced.
                      |   Would be two bytes longer (12->23) if object were referenced.
     22->.. fName     = Number of bytes in name of class that this TStreamerInfo object
                      |   describes, followed by the class name itself.  (TNamed::fName).
      0->.. fTitle    = Number of bytes in title of class that this TStreamerInfo object
                      |   describes, followed by the class title itself.  (TNamed::fTitle).
                      |  (Class title may be zero length)
 -End TNamed object
      0->3  fCheckSum = Check sum for class that this TStreamerInfo object describes.
                      |  This checksum is over all base classes and all persistent
                      |  non-static data members.  It is computed by TClass::GetCheckSum().
                      |  (TStreamerInfo::fCheckSum)
      4->7  fClassVersion = Version of class that this TStreamerInfo object describes.
                      |   (TStreamerInfo::fClassVersion)
 -Begin TObjArray object (Data member of TStreamerInfo)
      0->3  ByteCount = Number of remaining bytes in TObjArray object (uncompressed)
                      |   OR'd with kByteCountMask (0x40000000)
      4->.. ClassInfo = Information about TObjArray class
                      | If this is the first occurrence of a TObjArray object in the record
                      |  4->7  -1        = New class tag (constant kNewClassTag = 0xffffffff)
                      |  8->17 Classname = Object Class Name "TObjArray" (null terminated)
                      | Otherwise
                      |  4->7 clIdx      = Byte offset of new class tag in record, plus 2.
                      | OR'd with kClassMask (0x80000000)
      0->3  ByteCount = Number of remaining bytes in TObjArray object (uncompressed)
                      |   OR'd with kByteCountMask (0x40000000)
      4->5  Version   = Version of TObjArray Class
      6->15           = TObject object (a base class of TObjArray) (see \ref tobject).
                      |   Objects in StreamerInfo record are not referenced.
                      |   Would be two bytes longer (6->17) if object were referenced.
     16->16 fName     = Number of bytes in name of TObjArray object, followed by the
                      |   name itself.  (TCollection::fName).  TObjArray objects in
                      |   StreamerInfo record are unnamed, so byte contains 0.
     17->20 nObjects  = Number of objects (derived from TStreamerElement) in array.
     21->24 fLowerBound = Lower bound of array.  Will always be 0 in StreamerInfo record.
     25->.. objects   = Sequentially, TStreamerElement objects in the array.
                      | In a TStreamerInfo object, the objects in the TObjArray are
                      |   of various types (described below), all of which inherit
                      |   directly from TStreamerElement objects.  There will be one
                      |   such object for every base class of the class that the
                      |   TStreamerInfo object describes, and also one such object for
                      |   each persistent non-static data member of the class that the
                      |   TStreamerInfo object describes.
 -End TObjArray object and TStreamerInfo object
-------
</pre></div>

  The objects stored in the TObjectArray in TStreamerInfo are of various classes, each of
     which inherits directly from the TStreamerElement class.  The possible classes (which
     we refer to collectively as TStreamer<XXX>) are:

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
  TStreamerBase:          Used for a base class.  All others below used for data members.
  TStreamerBasicType:     For a basic type
  TStreamerString:        For type TString
  TStreamerBasicPointer:  For pointer to array of basic types
  TStreamerObject:        For an object derived from TObject
  TStreamerObjectPointer: For pointer to an object derived from TObject
  TStreamerLoop:          For pointer to an array of objects
  TStreamerObjectAny:     For an object not derived from TObject
  TStreamerSTL:           For an STL container (not yet used??)
  TStreamerSTLString:     For an STL string (not yet used??)
-------
</pre></div>

 Here is the format of a TStreamer<XXX> object in Release 3.02.06.
 In description below,

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
     0->3  ByteCount = Number of remaining bytes in TStreamer<XXX> object (uncompressed)
                     |   OR'd with kByteCountMask (0x40000000)
     4->.. ClassInfo = Information about the specific TStreamer<XXX> class
                     | If this is the first occurrence of a TStreamerXXX object in the record
                     |  4->7  -1        = New class tag (constant kNewClassTag = 0xffffffff)
                     |  8->.. Classname = Object Class Name "TStreamer<XXX>" (null terminated)
                     | Otherwise
                     |  4->7 clIdx      = Byte offset of new class tag in record, plus 2.
                    | OR'd with kClassMask (0x80000000)
     0->3  ByteCount = Number of remaining bytes in TStreamer<XXX> object (uncompressed)
                   |   OR'd with kByteCountMask (0x40000000)
     4->5  Version   = Version of TStreamer<XXX> Class
 -Begin TStreamerElement object (Base class of TStreamerXXX)
     0->3  ByteCount = Number of remaining bytes in TStreamerElement object (uncompressed)
                   |   OR'd with kByteCountMask (0x40000000)
     4->5  Version   = Version of TStreamerElement Class
 -Begin TNamed object (Base class of TStreamerElement)
     6->9  ByteCount = Number of remaining bytes in TNamed object
                     |   OR'd with kByteCountMask (0x40000000)
     10->11 Version   = Version of TNamed Class
     12->21           = TObject object (Base class of TNamed) (see \ref tobject).
                      |   Objects in StreamerInfo record are not referenced.
                      |   Would be two bytes longer (12->23) if object were referenced.
     22->.. fName     = Number of bytes in class name of base class or member name of
                      | data member that this TStreamerElement object describes,
                      | followed by the name itself. (TNamed::fName).
      0->.. fTitle    = Number of bytes in title of base class or data member that this
                      | TStreamerElement object describes, followed by the title itself.
                      |  (TNamed::fTitle).
 -End TNamed object
      0->3  fType     = Type of data described by this TStreamerElement.
                      |   (TStreamerElement::fType)
                      |   Built in types:
                      |   1:char, 2:short, 3:int, 4:long, 5:float, 8:double
                      |   11, 12, 13, 14:unsigned char, short, int, long respectively
                      |   6: an array dimension (counter)
                      |   15: bit mask (used for fBits field)
                      |
                      |   Pointers to built in types:
                      |   40 + fType of built in type (e.g. 43: pointer to int)
                      |
                      |   Objects:
                      |   65:TString, 66:TObject, 67:TNamed
                      |   0: base class (other than TObject or TNamed)
                      |   61: object data member derived from TObject (other than TObject or TNamed)
                      |   62: object data member not derived from TObject
                      |   63: pointer to object derived from TObject (pointer can't be null)
                      |   64: pointer to object derived from TObject (pointer may be null)
                      |   501: pointer to an array of objects
                      |   500: an STL string or container
                      |
                      |   Arrays:
                      |   20 + fType of array element (e.g. 23: array of int)
                      |
      4->7  fSize     = Size of built in type or of pointer to built in type. 0 otherwise.
                      |  (TStreamerElement::fSize).
      8->11 fArrayLength = Size of array (0 if not array)
                      |  (TStreamerElement::fArrayLength).
     12->15 fArrayDim = Number of dimensions of array (0 if not an array)
                      |  (TStreamerElement::fArrayDim).
     16->35 fMaxIndex = Five integers giving the array dimensions (0 if not applicable)
                      |  (TStreamerElement::fMaxIndex).
     36->.. fTypeName = Number of bytes in name of the data type of the data member that
                      |  the TStreamerElement object describes, followed by the name
                      |  itself.  If this TStreamerElement object defines a base class
                      |  rather than a data member, the name used is 'BASE'.
                      |  (TStreamerElement::fTypeName).
 -End TStreamerElement object
     The remaining data is specific to the type of TStreamer<XXX> class.
      For TStreamerInfoBase:
      0->3  fBaseVersion   = Version of base class that this TStreamerElement describes.
      For TStreamerBasicType:
            No specific data
      For TStreamerString:
            No specific data
      For TStreamerBasicPointer:
      0->3  fCountVersion = Version of class with the count (array dimension)
      4->.. fCountName= Number of bytes in the name of the data member holding
                      | the count, followed by the name itself.
      0->.. fCountName= Number of bytes in the name of the class holding the
                      | count, followed by the name itself.
      For TStreamerObject:
            No specific data
      For TStreamerObjectPointer:
            No specific data
      For TStreamerLoop:
      0->3  fCountVersion = Version of class with the count (array dimension)
      4->.. fCountName= Number of bytes in the name of the data member holding
                      | the count, followed by the name itself.
      0->.. fCountClass= Number of bytes in the name of the class holding the
                      | count, followed by the name itself.
      For TStreamerObjectAny:
            No specific data
      For TStreamerSTL:
      0->3  fSTLtype  = Type of STL container:
                      | 1:vector, 2:list, 3:deque, 4:map, 5:set, 6:multimap, 7:multiset
      4->7  fCType    = Type contained in STL container:
                      | Same values as for fType above, with one addition: 365:STL string
      For TStreamerSTLString:
            No specific data
</pre></div>
