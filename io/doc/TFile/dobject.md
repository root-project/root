\page dobject Format of a class object in DATA

### Release 3.02.06

Here is the format of a class object in DATA that uses the default streamer.
Objects of many classes with custom streamers can have very similar formats.

<div style="background-color: lightgrey; font-size: small;"><pre>
----------------
  0->3  ByteCount = Number of remaining bytes in object (uncompressed)
        | OR'd with kByteCountMask (0x40000000)
  4->.. ClassInfo = Information about class of object
        | If this is the first occurrence of an object of this class in the record
                       |  4->7  -1        = New class tag (constant kNewClassTag = 0xffffffff)
                       |  8->.. Classname = Object Class Name (null terminated string)
        | Otherwise
        |  4->7 clIdx      = Byte offset of new class tag in record, plus 2.
        | OR'd with kClassMask (0x80000000)
  0->3  ByteCount = Number of remaining bytes in object (uncompressed)
        | OR'd with kByteCountMask (0x40000000)
  4->5  Version   = Version of Class
</pre></div>

 The rest consists of objects of base classes and persistent non-static data members.
 Data members marked as transient are not stored.

<div style="background-color: lightgrey; font-size: small;"><pre>
  6->.. Sequentially, Objects of each base class from which this class is derived
    (rarely more than one)
  0->.. Sequentially, Objects of all non-static persistent data members.
</pre></div>

 Class objects are broken down recursively as above.

<div style="background-color: lightgrey; font-size: small;"><pre>
      Built in types are stored as follows:
 1 Byte: char, unsigned char
      2 Bytes: short, unsigned short
      4 Bytes: int, unsigned int, float
      8 Bytes: long, unsigned long, double
</pre></div>
Note that a long (signed or unsigned) is stored as 8 bytes even if it is only four bytes
in memory.  In that case, it is filled with leading zeros (or ones, for a negative value).

