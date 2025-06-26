import { R__unzip } from './io.mjs';
const LITTLE_ENDIAN = true;
class RBufferReader {

  constructor(buffer) {
    if (buffer instanceof ArrayBuffer) {
    this.buffer = buffer;
    this.byteOffset = 0;
    this.byteLength = buffer.byteLength;
    } else if (ArrayBuffer.isView(buffer)) {
    this.buffer = buffer.buffer;
    this.byteOffset = buffer.byteOffset;
    this.byteLength = buffer.byteLength;
    } else
      throw new TypeError('Invalid buffer type');

    this.view = new DataView(this.buffer);
    this.offset = 0;
  }

  // Move to a specific position in the buffer
  seek(position) {
    this.offset = position;
  }

  // Read unsigned 8-bit integer (1 BYTE)
  readU8() {
    const val = this.view.getUint8(this.offset);
    this.offset += 1;
    return val;
  }

  // Read unsigned 16-bit integer (2 BYTES)
  readU16() {
    const val = this.view.getUint16(this.offset, LITTLE_ENDIAN);
    this.offset += 2;
    return val;
  }

  // Read unsigned 32-bit integer (4 BYTES)
  readU32() {
    const val = this.view.getUint32(this.offset, LITTLE_ENDIAN);
    this.offset += 4;
    return val;
  }

  // Read signed 8-bit integer (1 BYTE)
  readS8() {
    const val = this.view.getInt8(this.offset);
    this.offset += 1;
    return val;
  }

  // Read signed 16-bit integer (2 BYTES)
  readS16() {
    const val = this.view.getInt16(this.offset, LITTLE_ENDIAN);
    this.offset += 2;
    return val;
  }

  // Read signed 32-bit integer (4 BYTES)
  readS32() {
    const val = this.view.getInt32(this.offset, LITTLE_ENDIAN);
    this.offset += 4;
    return val;
  }

  // Read 32-bit float (4 BYTES)
  readF32() {
    const val = this.view.getFloat32(this.offset, LITTLE_ENDIAN);
    this.offset += 4;
    return val;
  }

  // Read 64-bit float (8 BYTES)
  readF64() {
    const val = this.view.getFloat64(this.offset, LITTLE_ENDIAN);
    this.offset += 8;
    return val;
  }

  // Read a string with 32-bit length prefix
  readString() {
    const length = this.readU32();
    let str = '';
    for (let i = 0; i < length; i++)
      str += String.fromCharCode(this.readU8());
    return str;
  }

    // Read unsigned 64-bit integer (8 BYTES)
  readU64() {
    const val = this.view.getBigUint64(this.offset, LITTLE_ENDIAN);
    this.offset += 8;
    return val;
  }

  // Read signed 64-bit integer (8 BYTES)
  readS64() {
    const val = this.view.getBigInt64(this.offset, LITTLE_ENDIAN);
    this.offset += 8;
    return val;
  }

}


class RNTupleDescriptorBuilder {

deserializeHeader(header_blob) {
    if (!header_blob) return;

  const reader = new RBufferReader(header_blob),
 
  payloadStart = reader.offset,
  // Read the envelope metadata
  { envelopeLength } = this._readEnvelopeMetadata(reader),

  // Seek to end of envelope to get checksum
  checksumPos = payloadStart + envelopeLength - 8,
  currentPos = reader.offset;

  reader.seek(checksumPos);
  this.headerEnvelopeChecksum = reader.readU64(); 

  reader.seek(currentPos);

  //  Read feature flags list (may span multiple 64-bit words)
  this._readFeatureFlags(reader);

  //  Read metadata strings
  this.name = reader.readString();
  this.description = reader.readString();
  this.writer = reader.readString();

  // 4 list frames inside the header envelope
  this._readSchemaDescription(reader);
  }

deserializeFooter(footer_blob) {
    if (!footer_blob) return;

    const reader = new RBufferReader(footer_blob);

    // Read the envelope metadata
    this._readEnvelopeMetadata(reader);


    // Feature flag(32 bits)
    this._readFeatureFlags(reader);
    // Header checksum (64-bit xxhash3)
    const headerChecksumFromFooter = reader.readU64(); 
    if (headerChecksumFromFooter !== this.headerEnvelopeChecksum)
    throw new Error('RNTuple corrupted: header checksum does not match footer checksum.');

    const schemaExtensionSize = reader.readS64(); 

    console.log('Schema extension frame size:', schemaExtensionSize);
    if (schemaExtensionSize < 0)
      throw new Error('Schema extension frame is not a record frame, which is unexpected.');      
    
    // Schema extension record frame (4 list frames inside)
    this._readSchemaDescription(reader);

    // Cluster Group record frame
    this._readClusterGroups(reader);
  }


_readEnvelopeMetadata(reader) {
  const typeAndLength = reader.readU64(),

  // Envelope metadata
  // The 16 bits are the envelope type ID, and the 48 bits are the envelope length
  envelopeType = Number(typeAndLength & 0xFFFFn),
  envelopeLength = Number((typeAndLength >> 16n) & 0xFFFFFFFFFFFFn);

  console.log('Envelope Type ID:', envelopeType);
  console.log('Envelope Length:', envelopeLength);
  return { envelopeType, envelopeLength };
}

_readSchemaDescription(reader) {
  // Reading new descriptor arrays from the input
  const newFields = this._readFieldDescriptors(reader),
  newColumns = this._readColumnDescriptors(reader),
  newAliases = this._readAliasColumn(reader),
  newExtra = this._readExtraTypeInformation(reader);

  // Merging these new arrays into existing arrays
  this.fieldDescriptors = (this.fieldDescriptors || []).concat(newFields);
  this.columnDescriptors = (this.columnDescriptors || []).concat(newColumns);
  this.aliasColumns = (this.aliasColumns || []).concat(newAliases);
  this.extraTypeInfo = (this.extraTypeInfo || []).concat(newExtra);
}


_readFeatureFlags(reader) {
  this.featureFlags = [];
  while (true) {
    const val = reader.readU64();
    this.featureFlags.push(val);
    if ((val & 0x8000000000000000n) === 0n) break; // MSB not set: end of list
  }

  // verify all feature flags are zero
  if (this.featureFlags.some(v => v !== 0n))
  throw new Error('Unexpected non-zero feature flags: ' + this.featureFlags);
}

_readFieldDescriptors(reader) {
const fieldListSize = reader.readS64(), // signed 64-bit
fieldListIsList = fieldListSize < 0;


  if (!fieldListIsList)
    throw new Error('Field list frame is not a list frame, which is required.');

  const fieldListCount = reader.readU32(); // number of field entries
  console.log('Field List Count:', fieldListCount);

  // List frame: list of field record frames

  const fieldDescriptors = [];
  for (let i = 0; i < fieldListCount; ++i) {
    const fieldRecordSize = reader.readS64(),
    fieldVersion = reader.readU32(),
    typeVersion = reader.readU32(),
    parentFieldId = reader.readU32(),
    structRole = reader.readU16(),
    flags = reader.readU16(),

    fieldName = reader.readString(),
    typeName = reader.readString(),
    typeAlias = reader.readString(),
    description = reader.readString();
    console.log(`Field Record Size: ${fieldRecordSize}`);
    let arraySize = null, sourceFieldId = null, checksum = null;

    if (flags & 0x1) arraySize = reader.readU64();
    if (flags & 0x2) sourceFieldId = reader.readU32();
    if (flags & 0x4) checksum = reader.readU32();

     fieldDescriptors.push({
        fieldVersion,
        typeVersion,
        parentFieldId,
        structRole,
        flags,
        fieldName,
        typeName,
        typeAlias,
        description,
        arraySize,
        sourceFieldId,
        checksum
    });
}
  return fieldDescriptors;
}

_readColumnDescriptors(reader) {
  const columnListSize = reader.readS64(),
  columnListIsList = columnListSize < 0;
  if (!columnListIsList)
    throw new Error('Column list frame is not a list frame, which is required.');
  const columnListCount = reader.readU32(); // number of column entries
  console.log('Column List Count:', columnListCount);
  const columnDescriptors = [];
  for (let i = 0; i < columnListCount; ++i) {
  const columnRecordSize = reader.readS64(), 
  coltype = reader.readU16(),
  bitsOnStorage = reader.readU16(),
  fieldId = reader.readU32(),
  flags = reader.readU16(),
  representationIndex = reader.readU16();
  console.log(`Column Record Size: ${columnRecordSize}`);
   let firstElementIndex = null, minValue = null, maxValue = null;
  if (flags & 0x1) firstElementIndex = reader.readU64();
  if (flags & 0x2){
    minValue = reader.readF64();
    maxValue = reader.readF64();    
  }


  const column = {
      coltype,
      bitsOnStorage,
      fieldId,
      flags,
      representationIndex,
      firstElementIndex,
      minValue,
      maxValue
    };
    column.isDeferred = function() {
      return (this.flags & 0x01) !== 0;
    };
    column.isSuppressed = function() {
      return this.firstElementIndex !== null && this.firstElementIndex < 0;
    };

    columnDescriptors.push(column);
  }
 return columnDescriptors;
}
_readAliasColumn(reader){
  const aliasColumnListSize = reader.readS64(),
  aliasListisList = aliasColumnListSize < 0;
  if (!aliasListisList)
    throw new Error('Alias column list frame is not a list frame, which is required.');
  const aliasColumnCount = reader.readU32(); // number of alias column entries
  console.log('Alias Column List Count:', aliasColumnCount);
  const aliasColumns = [];
  for (let i = 0; i < aliasColumnCount; ++i){
  const aliasColumnRecordSize = reader.readS64(),
    physicalColumnId = reader.readU32(),
    fieldId = reader.readU32();
    console.log(`Alias Column Record Size: ${aliasColumnRecordSize}`);
    aliasColumns.push({
      physicalColumnId,
      fieldId
    });
  }
  return aliasColumns;
}
_readExtraTypeInformation(reader) {
  const extraTypeInfoListSize = reader.readS64(),
  isList = extraTypeInfoListSize < 0;

  if (!isList)
    throw new Error('Extra type info frame is not a list frame, which is required.');

  const entryCount = reader.readU32(); 
  console.log('Extra Type Info Count:', entryCount);

  const extraTypeInfo = [];
  for (let i = 0; i < entryCount; ++i) {
    const extraTypeInfoRecordSize = reader.readS64(),
    contentId = reader.readU32(),
    typeVersion = reader.readU32();
    console.log(`Extra Type Info Record Size: ${extraTypeInfoRecordSize}`);
    extraTypeInfo.push({
      contentId,
      typeVersion
    });
  }
  return extraTypeInfo;
}
_readClusterGroups(reader) {
  const clusterGroupListSize = reader.readS64(),
  isList = clusterGroupListSize < 0;
  if (!isList) throw new Error('Cluster group frame is not a list frame');

  const groupCount = reader.readU32();
  console.log('Cluster Group Count:', groupCount);

  const clusterGroups = [];

  for (let i = 0; i < groupCount; ++i) {
    const clusterRecordSize = reader.readS64(),
    minEntry = reader.readU64(),
    entrySpan = reader.readU64(),
    numClusters = reader.readU32();

    console.log(`Cluster Record Size: ${clusterRecordSize}`);
    console.log(`Min Entry: ${minEntry}, Entry Span: ${entrySpan}, Num Clusters: ${numClusters}`);

    clusterGroups.push({
      minEntry,
      entrySpan,
      numClusters,
    });
  }

  this.clusterGroups = clusterGroups;
}

}

/** @summary Very preliminary function to read header/footer from RNTuple
  * @private */
async function readHeaderFooter(tuple) {
   if (!tuple.$file)
      return false;

   // request header and footer buffers from the file
   return tuple.$file.readBuffer([tuple.fSeekHeader, tuple.fNBytesHeader, tuple.fSeekFooter, tuple.fNBytesFooter]).then(blobs => {
      if (blobs?.length !== 2)
         return false;

      // unzip both buffers
      return Promise.all([
         R__unzip(blobs[0], tuple.fLenHeader),
         R__unzip(blobs[1], tuple.fLenFooter)
      ]).then(unzip_blobs => {
         const header_blob = unzip_blobs[0],
               footer_blob = unzip_blobs[1];
         if (!header_blob || !footer_blob)
            return false;

         // create builder description and decode it - dummy for the moment

         tuple.builder = new RNTupleDescriptorBuilder;

         tuple.builder.deserializeHeader(header_blob);

         tuple.builder.deserializeFooter(footer_blob);

         return true;
      });
   });
}


/** @summary Create hierarchy of ROOT::RNTuple object
  * @desc Used by hierarchy painter to explore sub-elements
  * @private */
async function tupleHierarchy(tuple_node, tuple) {
   tuple_node._childs = [];
   // tuple_node._tuple = tuple;  // set reference, will be used later by RNTuple::Draw

   return readHeaderFooter(tuple).then(res => {
      if (!res)
         return res;

      tuple.builder?.fieldDescriptors.forEach(field => {
          tuple_node._childs.push({
            _name: field.fieldName,
            _kind: 'ROOT::RNTupleField', // pseudo class name, used in draw.mjs
            _title: `Filed of type ${field.typeName}`,
            _obj: field
         });
      });

      return true;
   });
}

export { tupleHierarchy, readHeaderFooter, RBufferReader };
