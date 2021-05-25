# RNTuple Reference Specifications (DRAFT, WIP)

## Terminology

### Column
A set of data values, all of a single fundamental type, which is a storage-backed array.

### Page 
Partitioned columns, typically a few tens of kilobytes in size. The smallest addresable unit.

### Cluster
A set of pages that contain all the data of a certain entry range. They are typically a few tens of megabytes
in size and a natural unit of processing for a single thread or task.

### Collection
Vector that allows for variable-length storage of elements (which can be arbitrarily complex, e.g., collection of record or collection of collection) within a row.

### Record
A collection of fields, possibly of different data types, typically in a fixed number and sequence, and containing variable-length collections of sub records.

### Header
Container for RNTuple meta-data relating to the the RNTuple schema.

### Footer
Container for RNTuple meta-data relating to the locations of the pages.

### Checkpoint footer
A feature extended to meta-data that allows for data recovery in case of an application crash during data taking.

### Logical Layer / C++ objects
Layer that splits objects into columns of fundamental types, thereby mapping the types onto columns.

### Primitives layer / simple types
Layer that governs the pool of uncompressed and deserialized pages in memory and the representation of fundamental types on disk.

### Storage layer / byte ranges
Layer that provides access to the byte ranges containing a page on a physical or virtual device. The storage layer manages compression and reads and writes to and from the I/O device. It also allocates memory for pages in order to allow for direct page mappings.

## Putting it all together: LEGO analogy
Imagine you are the LEGO company and are shipping out LEGO pieces for 2 identical spaceships to 2 different households: 2 spaceships for House A and 2 spaceships for House B. Each spaceship requires the same 6 red pieces, 4 yellow pieces, and 2 blue pieces for complete assembly.

In this ananlogy, our spaceship is the **entry (row)**. Similarly, the category of colored pieces are our **columns**. In other words, we have 3 total **columns** which consist of: red (which has 2 x 6 x 2 = 24 total red pieces), yellow (which has 2 x 4 x 2 = 16 total yellow pieces), and blue (which has 2 x 2 x 2 = 8 total blue pieces). Simiarly, a **page** is a plastic bag that can only hold 4 LEGO pieces. Furthermore, the **cluster** is the cardboard box that holds all the pieces (via the plastics bags, i.e. the pages) required to construct a single  spaceship. Upon receipt of the box's delivery to each household, each house will then assemble their single spaceship.

![lego analogy visualization](https://user-images.githubusercontent.com/35269716/109181886-6e867680-775a-11eb-8d45-3c5c747cbb39.png)

# Schema type system
This is a set of base types that we will use repeatedly for the actual header footer content. For example, we describe integer type and string type. This is contrast to other kinds of type systems, such as those used with the RNTuple payload or the ROOT TFile container. 

## Integer type
This type is used for the Header's meta-data categories, such as: TimeStampData, TimeStampWritten, and Version. It uses two's complement representation and follows Little Endian conventions. 

## String type
This type is used for the Header's meta-data categories, such as: Name, Description, Author, and Custodian. It applies the ASCII charter encoding standard (8 bits or 1 byte per character).

The first 32 bits (4 bytes) holds the length of the proceeding string. After this initial value is the array of characters of the string. Consequently, the minimum size of this type is 32 bits (4 bytes), even if there is no string for a particular Header meta-data field. 

## Header
The header stores RNTuple meta-data relating to its schema. The following meta-data list follows ROOT naming convention that prefixes each field name with an "f" in order to indicate that it is a member variable of a class ("f" stands for "field"):

### Name
The primary identifier of the ntuple. It follows the String type standards.

### Description
Explanation of what is in the ntuple and how it it is structured, etc. It follows the String type standards.

### Author
The original creator of the ntuple. It follows the String type standards.

### Custodian
The keeper of the ntuple, who maintains and stores the actual data. In contrast with the "Author," who created the data, the "Custodian" manages the integrity of the bits. It follows the String type standards.

### TimeStampData
The time that the ntuple was created. It follows the Integer type standards.

### TimeStampWritten
The time that the ntuple was written. In contrast with "TimeStampData," is used in copying situations. Therefore, by default the initial value is the same as TimeStampData (i.e., no duplication has yet occured), but will change as copying occurs. It follows the Integer type standards.

### Version
The specific version with respect to the data format (as opposed to the ROOT software version), i.e., the schema fields, header, and footer. There are two numbers: the first number is the version layout, and the second number is the minimum version number that the reader has to support in order to be able to use the data. This helps with backward and forward compatiblity.

### OwnUuid
A universally unique identifier (as defined by RFC4122) that is used to identify information in computer systems. Gives every generated nTuple a unique identifier. [Type to be determined].

### GroupUuid
The identifier used to identify a group of RNTuples that belong to each other (for instance, when an ntuple is written out as a result of processing another ntuple). Aside from this pluralistic trait, it possesses the same characteristics described in the field, "OwnUuid."


### FieldDescriptors
These hold information about each field, which is structured as list of structs. Each struct has key-value pair types, which includes the following:

#### FieldID
The identifying number for the respective field. It follows the Integer type standards.

#### FieldVersion
The specific version with respect to the mapping from the RNTuple logical layer (i.e. fields) to the RNTuple primitives layer (i.e. columns). There are two numbers: the first number is the version as written, and the second number is the minimum version number that the reader has to support in order to be able to use the data.

#### TypeVersion
The specific version with respect to the memory layout of the C++ type that corresponds to the field.

#### FieldName
The primary identifier of the field. It follows the String type standards.

#### FieldDescription
Explanation of what is in the field and how it it is structured, etc. It follows the String type standards.

#### TypeName
The C++ name of the type. It follows the String type standards.

#### NRepetitions
Numerical value of the fixed sized array configuration option. It follows the Integer type standards.

#### Structure
Numerical value(s) that reflect the broader organization of the data hierarchy. It follows the Integer type standards.

#### ParentId
The identification number of the parent field. It follows the Integer type standards.

#### LinkIds
A list of identifiers of the columns, which store the content of the field. It follows the Integer type standards.

### ColumnDescriptors
These hold information about each column, which is structured as a list of structs. Each struct has key-value pair types, which includes the following:

#### ColumnID
The identifying number for the respective column. It follows the Integer type standards.

#### ColumnModel
This consists of two numericals values: the first numerical value consists of either zero or one, which reflects whether or not the data in the column is sorted. The second numerical value matches a value from the following list of finite values which reflect the fundamental types:
   * kUnknown = 0
   * kIndex
   * kSwitch
   * kByte
   * kBit
   * kReal64
   * kReal32
   * kReal16
   * kReal8
   * kInt64
   * kInt32
   * kInt16

#### ColumnFieldID
The identifying number for the field that this column belongs to. It follows the Integer type standards.

#### ColumnIndex
Due to the fact that a field can have several columns attached to it, this numerical value specifies the column order. It follows the Integer type standards.


## Footer
The footers holds information such as the descriptors for each cluster. The following list goes into further detail regarding each container of information:

### ClusterDescriptors
These provide information regarding the location of the clusters and of the pages inside the clusters. To continue with the earlier LEGO analogy, this would contain information such as the range of "plastic bags" (pages).

### OwnUuid
A universally unique identifier (as defined by RFC4122) that is used to identify information in computer systems. Gives every generated nTuple a unique identifier. [Type to be determined].



