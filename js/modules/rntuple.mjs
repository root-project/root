import { isStr, isObject } from './core.mjs';
import { R__unzip } from './io.mjs';
import { TDrawSelector, treeDraw } from './tree.mjs';

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
      // important - offset should start from actual place in the buffer
      this.offset = this.byteOffset;
   }

   // Move to a specific position in the buffer
   seek(position) {
      if (typeof position === 'bigint') {
         if (position > BigInt(Number.MAX_SAFE_INTEGER))
            throw new Error(`Offset too large to seek safely: ${position}`);
         this.offset = Number(position);
      } else
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

const ENTupleColumnType = {
   kBit: 0x00,
   kByte: 0x01,
   kChar: 0x02,
   kInt8: 0x03,
   kUInt8: 0x04,
   kInt16: 0x05,
   kUInt16: 0x06,
   kInt32: 0x07,
   kUInt32: 0x08,
   kInt64: 0x09,
   kUInt64: 0x0A,
   kReal16: 0x0B,
   kReal32: 0x0C,
   kReal64: 0x0D,
   kIndex32: 0x0E,
   kIndex64: 0x0F,
   kSwitch: 0x10,
   kSplitInt16: 0x11,
   kSplitUInt16: 0x12,
   kSplitInt32: 0x13,
   kSplitUInt32: 0x14,
   kSplitInt64: 0x15,
   kSplitUInt64: 0x16,
   kSplitReal16: 0x17,
   kSplitReal32: 0x18,
   kSplitReal64: 0x19,
   kSplitIndex32: 0x1A,
   kSplitIndex64: 0x1B,
   kReal32Trunc: 0x1C,
   kReal32Quant: 0x1D
};


/**
 * @summary Rearrange bytes from split format to normal format (row-wise) for decoding
 */
function recontructUnsplitBuffer(blob, columnDescriptor) {
   const { coltype } = columnDescriptor;

   if (
      coltype === ENTupleColumnType.kSplitUInt16 ||
        coltype === ENTupleColumnType.kSplitUInt32 ||
        coltype === ENTupleColumnType.kSplitUInt64 ||
        coltype === ENTupleColumnType.kSplitReal16 ||
        coltype === ENTupleColumnType.kSplitReal32 ||
        coltype === ENTupleColumnType.kSplitReal64 ||
        coltype === ENTupleColumnType.kSplitIndex32 ||
        coltype === ENTupleColumnType.kSplitIndex64 ||
        coltype === ENTupleColumnType.kSplitInt16 ||
        coltype === ENTupleColumnType.kSplitInt32 ||
        coltype === ENTupleColumnType.kSplitInt64
   ) {
      // Determine byte size based on column type
      let byteSize;
      switch (coltype) {
         case ENTupleColumnType.kSplitReal64:
         case ENTupleColumnType.kSplitInt64:
         case ENTupleColumnType.kSplitUInt64:
         case ENTupleColumnType.kSplitIndex64:
            byteSize = 8;
            break;
         case ENTupleColumnType.kSplitReal32:
         case ENTupleColumnType.kSplitInt32:
         case ENTupleColumnType.kSplitIndex32:
         case ENTupleColumnType.kSplitUInt32:
            byteSize = 4;
            break;
         case ENTupleColumnType.kSplitInt16:
         case ENTupleColumnType.kSplitUInt16:
         case ENTupleColumnType.kSplitReal16:
            byteSize = 2;
            break;
         default:
            throw new Error(`Unsupported split coltype: ${coltype} (0x${coltype.toString(16).padStart(2, '0')})`);
      }

      const splitView = new DataView(blob.buffer, blob.byteOffset, blob.byteLength),
            count = blob.byteLength / byteSize,
            outBuffer = new ArrayBuffer(blob.byteLength),
            outBytes = new Uint8Array(outBuffer);

      for (let i = 0; i < count; ++i) {
         for (let b = 0; b < byteSize; ++b) {
            const splitIndex = b * count + i,
                  byte = splitView.getUint8(splitIndex),
                  writeIndex = i * byteSize + b;
            outBytes[writeIndex] = byte;
         }
      }

      // Return updated blob and remapped coltype
      const newBlob = outBuffer;
      let newColtype;
      switch (coltype) {
         case ENTupleColumnType.kSplitUInt16:
            newColtype = ENTupleColumnType.kUInt16;
            break;
         case ENTupleColumnType.kSplitUInt32:
            newColtype = ENTupleColumnType.kUInt32;
            break;
         case ENTupleColumnType.kSplitUInt64:
            newColtype = ENTupleColumnType.kUInt64;
            break;
         case ENTupleColumnType.kSplitIndex32:
            newColtype = ENTupleColumnType.kIndex32;
            break;
         case ENTupleColumnType.kSplitIndex64:
            newColtype = ENTupleColumnType.kIndex64;
            break;
         case ENTupleColumnType.kSplitReal16:
            newColtype = ENTupleColumnType.kReal16;
            break;
         case ENTupleColumnType.kSplitReal32:
            newColtype = ENTupleColumnType.kReal32;
            break;
         case ENTupleColumnType.kSplitReal64:
            newColtype = ENTupleColumnType.kReal64;
            break;
         case ENTupleColumnType.kSplitInt16:
            newColtype = ENTupleColumnType.kInt16;
            break;
         case ENTupleColumnType.kSplitInt32:
            newColtype = ENTupleColumnType.kInt32;
            break;
         case ENTupleColumnType.kSplitInt64:
            newColtype = ENTupleColumnType.kInt64;
            break;
         default:
            throw new Error(`Unsupported split coltype for reassembly: ${coltype}`);
      }

      return { blob: newBlob, coltype: newColtype };
   }

   // If no split type, return original blob and coltype
   return { blob, coltype };
}


/**
 * @summary Decode a reconstructed index buffer (32- or 64-bit deltas to absolute indices)
 */
function DecodeDeltaIndex(blob, coltype) {
   let deltas, result;

   if (coltype === ENTupleColumnType.kIndex32) {
      deltas = new Int32Array(blob.buffer || blob, blob.byteOffset || 0, blob.byteLength / 4);
      result = new Int32Array(deltas.length);
   } else if (coltype === ENTupleColumnType.kIndex64) {
      deltas = new BigInt64Array(blob.buffer || blob, blob.byteOffset || 0, blob.byteLength / 8);
      result = new BigInt64Array(deltas.length);
   } else
      throw new Error(`DecodeDeltaIndex: unsupported column type ${coltype}`);

   if (deltas.length > 0)
      result[0] = deltas[0];
   for (let i = 1; i < deltas.length; ++i)
      result[i] = result[i - 1] + deltas[i];

   return { blob: result, coltype };
}

/**
 * @summary Decode a reconstructed signed integer buffer using ZigZag encoding
  */
function decodeZigzag(blob, coltype) {
   let zigzag, result;

   if (coltype === ENTupleColumnType.kInt16) {
      zigzag = new Uint16Array(blob.buffer || blob, blob.byteOffset || 0, blob.byteLength / 2);
      result = new Int16Array(zigzag.length);
   } else if (coltype === ENTupleColumnType.kInt32) {
      zigzag = new Uint32Array(blob.buffer || blob, blob.byteOffset || 0, blob.byteLength / 4);
      result = new Int32Array(zigzag.length);
   } else if (coltype === ENTupleColumnType.kInt64) {
      zigzag = new BigUint64Array(blob.buffer || blob, blob.byteOffset || 0, blob.byteLength / 8);
      result = new BigInt64Array(zigzag.length);
   } else
      throw new Error(`decodeZigzag: unsupported column type ${coltype}`);

   for (let i = 0; i < zigzag.length; ++i) {
      // ZigZag decode: (x >>> 1) ^ (-(x & 1))
      const x = zigzag[i];
      result[i] = (x >>> 1) ^ (-(x & 1));
   }

   return { blob: result, coltype };
}

// Envelope Types
// TODO: Define usage logic for envelope types in future
// const kEnvelopeTypeHeader = 0x01,
//       kEnvelopeTypeFooter = 0x02,
//       kEnvelopeTypePageList = 0x03,

// Field Flags
const kFlagRepetitiveField = 0x01,
      kFlagProjectedField = 0x02,
      kFlagHasTypeChecksum = 0x04,

   // Column Flags
      kFlagDeferredColumn = 0x01,
      kFlagHasValueRange = 0x02;

class RNTupleDescriptorBuilder {

   deserializeHeader(header_blob) {
      if (!header_blob)
         return;

      const reader = new RBufferReader(header_blob),

            payloadStart = reader.offset,
            // Read the envelope metadata
            {
               envelopeLength
            } = this._readEnvelopeMetadata(reader),

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
      if (!footer_blob)
         return;

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

      return {
         envelopeType,
         envelopeLength
      };
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
         if ((val & 0x8000000000000000n) === 0n)
            break; // MSB not set: end of list
      }

      // verify all feature flags are zero
      if (this.featureFlags.some(v => v !== 0n))
         throw new Error('Unexpected non-zero feature flags: ' + this.featureFlags);
   }

   _readFieldDescriptors(reader) {
      const startOffset = BigInt(reader.offset),
            fieldListSize = reader.readS64(), // signed 64-bit
            fieldListIsList = fieldListSize < 0;


      if (!fieldListIsList)
         throw new Error('Field list frame is not a list frame, which is required.');

      const fieldListCount = reader.readU32(), // number of field entries
         // List frame: list of field record frames

            fieldDescriptors = [];
      for (let i = 0; i < fieldListCount; ++i) {
         const recordStart = BigInt(reader.offset),
               fieldRecordSize = reader.readS64(),
               fieldVersion = reader.readU32(),
               typeVersion = reader.readU32(),
               parentFieldId = reader.readU32(),
               structRole = reader.readU16(),
               flags = reader.readU16(),

               fieldName = reader.readString(),
               typeName = reader.readString(),
               typeAlias = reader.readString(),
               description = reader.readString();
         let arraySize = null,
             sourceFieldId = null,
             checksum = null;

         if (flags & kFlagRepetitiveField)
            arraySize = reader.readU64();

         if (flags & kFlagProjectedField)
            sourceFieldId = reader.readU32();

         if (flags & kFlagHasTypeChecksum)
            checksum = reader.readU32();


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
         reader.seek(Number(recordStart + fieldRecordSize));
      }
      reader.seek(Number(startOffset - fieldListSize));
      return fieldDescriptors;
   }

   _readColumnDescriptors(reader) {
      const startOffset = BigInt(reader.offset),
            columnListSize = reader.readS64(),
            columnListIsList = columnListSize < 0;
      if (!columnListIsList)
         throw new Error('Column list frame is not a list frame, which is required.');
      const columnListCount = reader.readU32(), // number of column entries
            columnDescriptors = [];
      for (let i = 0; i < columnListCount; ++i) {
         const recordStart = BigInt(reader.offset),
               columnRecordSize = reader.readS64(),
               coltype = reader.readU16(),
               bitsOnStorage = reader.readU16(),
               fieldId = reader.readU32(),
               flags = reader.readU16(),
               representationIndex = reader.readU16();
         let firstElementIndex = null,
             minValue = null,
             maxValue = null;

         if (flags & kFlagDeferredColumn)
            firstElementIndex = reader.readU64();

         if (flags & kFlagHasValueRange) {
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
            maxValue,
            index: i
         };
         column.isDeferred = function() {
            return (this.flags & RNTupleDescriptorBuilder.kFlagDeferredColumn) !== 0;
         };
         column.isSuppressed = function() {
            return this.firstElementIndex !== null && this.firstElementIndex < 0;
         };

         columnDescriptors.push(column);
         reader.seek(Number(recordStart + columnRecordSize));
      }
      reader.seek(Number(startOffset - columnListSize));
      return columnDescriptors;
   }
   _readAliasColumn(reader) {
      const startOffset = BigInt(reader.offset),
            aliasColumnListSize = reader.readS64(),
            aliasListisList = aliasColumnListSize < 0;
      if (!aliasListisList)
         throw new Error('Alias column list frame is not a list frame, which is required.');
      const aliasColumnCount = reader.readU32(), // number of alias column entries
            aliasColumns = [];
      for (let i = 0; i < aliasColumnCount; ++i) {
         const recordStart = BigInt(reader.offset),
               aliasColumnRecordSize = reader.readS64(),
               physicalColumnId = reader.readU32(),
               fieldId = reader.readU32();
         aliasColumns.push({
            physicalColumnId,
            fieldId
         });
         reader.seek(Number(recordStart + aliasColumnRecordSize));
      }
      reader.seek(Number(startOffset - aliasColumnListSize));
      return aliasColumns;
   }
   _readExtraTypeInformation(reader) {
      const startOffset = BigInt(reader.offset),
            extraTypeInfoListSize = reader.readS64(),
            isList = extraTypeInfoListSize < 0;

      if (!isList)
         throw new Error('Extra type info frame is not a list frame, which is required.');

      const entryCount = reader.readU32(),

            extraTypeInfo = [];
      for (let i = 0; i < entryCount; ++i) {
         const recordStart = BigInt(reader.offset),
               extraTypeInfoRecordSize = reader.readS64(),
               contentId = reader.readU32(),
               typeVersion = reader.readU32();
         extraTypeInfo.push({
            contentId,
            typeVersion
         });
         reader.seek(Number(recordStart + extraTypeInfoRecordSize));
      }
      reader.seek(Number(startOffset - extraTypeInfoListSize));
      return extraTypeInfo;
   }
   _readClusterGroups(reader) {
      const startOffset = BigInt(reader.offset),
            clusterGroupListSize = reader.readS64(),
            isList = clusterGroupListSize < 0;
      if (!isList)
         throw new Error('Cluster group frame is not a list frame');

      const groupCount = reader.readU32(),

            clusterGroups = [];

      for (let i = 0; i < groupCount; ++i) {
         const recordStart = BigInt(reader.offset),
               clusterRecordSize = reader.readS64(),
               minEntry = reader.readU64(),
               entrySpan = reader.readU64(),
               numClusters = reader.readU32(),
               pageListLength = reader.readU64(),


            // Locator method to get the page list locator offset
               pageListLocator = this._readLocator(reader),


               group = {
                  minEntry,
                  entrySpan,
                  numClusters,
                  pageListLocator,
                  pageListLength
               };
         clusterGroups.push(group);
         reader.seek(Number(recordStart + clusterRecordSize));
      }
      reader.seek(Number(startOffset - clusterGroupListSize));
      this.clusterGroups = clusterGroups;
   }

   _readLocator(reader) {
      const sizeAndType = reader.readU32(); // 4 bytes: size + T bit
      if ((sizeAndType | 0) < 0) // | makes the sizeAndType as signed
         throw new Error('Non-standard locators (T=1) not supported yet');
      const size = sizeAndType,
            offset = reader.readU64(); // 8 bytes: offset
      return {
         size,
         offset
      };
   }
   deserializePageList(page_list_blob) {
      if (!page_list_blob)
         throw new Error('deserializePageList: received an invalid or empty page list blob');

      const reader = new RBufferReader(page_list_blob);
      this._readEnvelopeMetadata(reader);
      // Page list checksum (64-bit xxhash3)
      const pageListHeaderChecksum = reader.readU64();
      if (pageListHeaderChecksum !== this.headerEnvelopeChecksum)
         throw new Error('RNTuple corrupted: header checksum does not match Page List Header checksum.');

      const listStartOffset = BigInt(reader.offset),
            // Read cluster summaries list frame
            clusterSummaryListSize = reader.readS64();
      if (clusterSummaryListSize >= 0)
         throw new Error('Expected a list frame for cluster summaries');
      const clusterSummaryCount = reader.readU32(),

            clusterSummaries = [];

      for (let i = 0; i < clusterSummaryCount; ++i) {
         const recordStart = BigInt(reader.offset),
               clusterSummaryRecordSize = reader.readS64(),
               firstEntry = reader.readU64(),
               combined = reader.readU64(),
               flags = combined >> 56n;
         if (flags & 0x01n)
            throw new Error('Cluster summary uses unsupported sharded flag (0x01)');
         const numEntries = Number(combined & 0x00FFFFFFFFFFFFFFn);
         clusterSummaries.push({
            firstEntry,
            numEntries,
            flags
         });
         reader.seek(Number(recordStart + clusterSummaryRecordSize));
      }
      reader.seek(Number(listStartOffset - clusterSummaryListSize));
      this.clusterSummaries = clusterSummaries;
      this._readNestedFrames(reader);

      /* const checksumPagelist = */ reader.readU64();
   }

   _readNestedFrames(reader) {
      const clusterPageLocations = [],
            numListClusters = reader.readS64();
      if (numListClusters >= 0)
         throw new Error('Expected list frame for clusters');
      const numRecordCluster = reader.readU32();

      for (let i = 0; i < numRecordCluster; ++i) {
         const outerListSize = reader.readS64();
         if (outerListSize >= 0)
            throw new Error('Expected outer list frame for columns');

         const numColumns = reader.readU32(),
               columns = [];

         for (let c = 0; c < numColumns; ++c) {
            const innerListSize = reader.readS64();
            if (innerListSize >= 0)
               throw new Error('Expected inner list frame for pages');

            const numPages = reader.readU32(),
                  pages = [];

            for (let p = 0; p < numPages; ++p) {
               const numElementsWithBit = reader.readS32(),
                     hasChecksum = numElementsWithBit < 0,
                     numElements = BigInt(Math.abs(Number(numElementsWithBit))),

                     locator = this._readLocator(reader);
               pages.push({
                  numElements,
                  hasChecksum,
                  locator
               });
            }

            const elementOffset = reader.readS64(),
                  isSuppressed = elementOffset < 0;

            let compression = null;
            if (!isSuppressed)
               compression = reader.readU32();

            columns.push({
               pages,
               elementOffset,
               isSuppressed,
               compression
            });
         }

         clusterPageLocations.push(columns);
      }

      this.pageLocations = clusterPageLocations;
   }

   // Example Of Deserializing Page Content
   deserializePage(blob, columnDescriptor, pageInfo) {
      const originalColtype = columnDescriptor.coltype,
            {
               coltype
            } = recontructUnsplitBuffer(blob, columnDescriptor);
      let {
         blob: processedBlob
      } = recontructUnsplitBuffer(blob, columnDescriptor);


      // Handle split index types
      if (originalColtype === ENTupleColumnType.kSplitIndex32 || originalColtype === ENTupleColumnType.kSplitIndex64) {
         const {
            blob: decodedArray
         } = DecodeDeltaIndex(processedBlob, coltype);
         processedBlob = decodedArray;
      }

      // Handle Split Signed Int types
      if (originalColtype === ENTupleColumnType.kSplitInt16 || originalColtype === ENTupleColumnType.kSplitInt32 || originalColtype === ENTupleColumnType.kSplitInt64) {
         const {
            blob: decodedArray
         } = decodeZigzag(processedBlob, coltype);
         processedBlob = decodedArray;
      }

      const reader = new RBufferReader(processedBlob),
            values = [],

            // Use numElements from pageInfo parameter
            numValues = Number(pageInfo.numElements),
            // Helper for all simple types
            extractValues = (readFunc) => {
               for (let i = 0; i < numValues; ++i)
                  values.push(readFunc());
            };
      switch (coltype) {
         case ENTupleColumnType.kBit: {
            let bitCount = 0;
            const totalBitsInBuffer = processedBlob.byteLength * 8;
            if (totalBitsInBuffer < numValues)
               throw new Error(`kBit: Not enough bits in buffer (${totalBitsInBuffer}) for numValues (${numValues})`);

            for (let byteIndex = 0; byteIndex < processedBlob.byteLength; ++byteIndex) {
               const byte = reader.readU8();

               // Extract 8 bits from this byte
               for (let bitPos = 0; bitPos < 8 && bitCount < numValues; ++bitPos, ++bitCount) {
                  const bitValue = (byte >>> bitPos) & 1,
                        boolValue = bitValue === 1;
                  values.push(boolValue);
               }
            }
            break;
         }

         case ENTupleColumnType.kReal64:
            extractValues(reader.readF64.bind(reader));
            break;
         case ENTupleColumnType.kReal32:
            extractValues(reader.readF32.bind(reader));
            break;
         case ENTupleColumnType.kInt64:
            extractValues(reader.readS64.bind(reader));
            break;
         case ENTupleColumnType.kUInt64:
            extractValues(reader.readU64.bind(reader));
            break;
         case ENTupleColumnType.kInt32:
            extractValues(reader.readS32.bind(reader));
            break;
         case ENTupleColumnType.kUInt32:
            extractValues(reader.readU32.bind(reader));
            break;
         case ENTupleColumnType.kInt16:
            extractValues(reader.readS16.bind(reader));
            break;
         case ENTupleColumnType.kUInt16:
            extractValues(reader.readU16.bind(reader));
            break;
         case ENTupleColumnType.kInt8:
            extractValues(reader.readS8.bind(reader));
            break;
         case ENTupleColumnType.kUInt8:
         case ENTupleColumnType.kByte:
            extractValues(reader.readU8.bind(reader));
            break;
         case ENTupleColumnType.kChar:
            extractValues(() => String.fromCharCode(reader.readS8()));
            break;
         case ENTupleColumnType.kIndex32:
            extractValues(reader.readS32.bind(reader));
            break;
         case ENTupleColumnType.kIndex64:
            extractValues(reader.readS64.bind(reader));
            break;
         default:
            throw new Error(`Unsupported column type: ${columnDescriptor.coltype}`);
      }
      return values;
   }

} // class RNTupleDescriptorBuilder


/** @summary Very preliminary function to read header/footer from RNTuple
 * @private */
async function readHeaderFooter(tuple) {
   // if already read - return immediately, make possible to call several times
   if (tuple?.builder)
      return true;

   if (!tuple.$file)
      return false;

   // request header and footer buffers from the file
   return tuple.$file.readBuffer([tuple.fSeekHeader, tuple.fNBytesHeader, tuple.fSeekFooter, tuple.fNBytesFooter]).then(blobs => {
      if (blobs?.length !== 2)
         return false;

      // Handle both compressed and uncompressed cases
      const processBlob = (blob, uncompressedSize) => {
         // If uncompressedSize matches blob size, it's uncompressed
         if (blob.byteLength === uncompressedSize)
            return Promise.resolve(blob);
         return R__unzip(blob, uncompressedSize);
      };

      return Promise.all([
         processBlob(blobs[0], tuple.fLenHeader),
         processBlob(blobs[1], tuple.fLenFooter)
      ]).then(unzip_blobs => {
         const [header_blob, footer_blob] = unzip_blobs;
         if (!header_blob || !footer_blob)
            return false;

         tuple.builder = new RNTupleDescriptorBuilder;
         tuple.builder.deserializeHeader(header_blob);
         tuple.builder.deserializeFooter(footer_blob);

         // Build fieldToColumns mapping
         tuple.fieldToColumns = {};
         for (const colDesc of tuple.builder.columnDescriptors) {
            const fieldDesc = tuple.builder.fieldDescriptors[colDesc.fieldId],
                  fieldName = fieldDesc.fieldName;
            if (!tuple.fieldToColumns[fieldName])
               tuple.fieldToColumns[fieldName] = [];
            tuple.fieldToColumns[fieldName].push(colDesc);
         }

         // Deserialize Page List
         const group = tuple.builder.clusterGroups?.[0];
         if (!group || !group.pageListLocator)
            throw new Error('No valid cluster group or page list locator found');

         const offset = Number(group.pageListLocator.offset),
               size = Number(group.pageListLocator.size),
               uncompressedSize = Number(group.pageListLength);

         return tuple.$file.readBuffer([offset, size]).then(page_list_blob => {
            if (!(page_list_blob instanceof DataView))
               throw new Error(`Expected DataView from readBuffer, got ${Object.prototype.toString.call(page_list_blob)}`);

            // Check if page list data is uncompressed
            if (page_list_blob.byteLength === uncompressedSize) {
               // Data is uncompressed, use directly
               tuple.builder.deserializePageList(page_list_blob);
               return true;
            }
            // Attempt to decompress the page list
            return R__unzip(page_list_blob, uncompressedSize).then(unzipped_blob => {
               if (!(unzipped_blob instanceof DataView))
                  throw new Error(`Unzipped page list is not a DataView, got ${Object.prototype.toString.call(unzipped_blob)}`);

               tuple.builder.deserializePageList(unzipped_blob);
               return true;
            });
         });
      });
   }).catch(err => {
      console.error('Error during readHeaderFooter execution:', err);
      throw err;
   });
}

function readEntry(rntuple, fieldName, clusterIndex, entryIndex) {
   const builder = rntuple.builder,
         field = builder.fieldDescriptors.find(f => f.fieldName === fieldName),
         columns = rntuple.fieldToColumns[fieldName];

   if (!field)
      throw new Error(`No descriptor for field ${fieldName}`);
   if (!columns)
      throw new Error(`No columns field ${fieldName}`);


   const pages = builder.pageLocations[clusterIndex]?.[columns[0].index]?.pages;
   if (!pages)
      throw new Error(`No pages found ${fieldName}`);

   let pageid = 0;
   while ((pageid < pages.length - 1) && (entryIndex >= Number(pages[pageid].numElements))) {
      entryIndex -= Number(pages[pageid].numElements);
      pageid++;
   }

   if (field.typeName === 'std::string') {
      // string extracted from two columns
      const offsets = rntuple._clusterData[columns[0].index][pageid],
            payload = rntuple._clusterData[columns[1].index][pageid],
            start = entryIndex === 0 ? 0 : Number(offsets[entryIndex - 1]),
            end = Number(offsets[entryIndex]);
      return payload.slice(start, end).join(''); // Convert to string
   }
   const values = rntuple._clusterData[columns[0].index];
   return values[pageid][entryIndex];
}

/** @summary Return field name for specified branch index
 * @desc API let use field name in selector or field object itself */
function getSelectorFieldName(selector, i) {
   const br = selector.getBranch(i);
   return isStr(br) ? br : br?.fieldName;
}

// Read and process the next data cluster from the RNTuple
async function readNextCluster(rntuple, selector) {
   const builder = rntuple.builder;

   // Add validation
   if (!builder.clusterSummaries)
      throw new Error('No cluster summaries available - possibly incomplete file reading');

   const clusterIndex = selector.currentCluster,
         clusterSummary = builder.clusterSummaries[clusterIndex],
      // Gather all pages for this cluster from selected fields only
         pages = [],
      // Collect only selected field names from selector
         selectedFields = [];

   if (!clusterSummary) {
      selector.Terminate(clusterIndex > 0);
      return false;
   }

   for (let i = 0; i < selector.numBranches(); ++i)
      selectedFields.push(getSelectorFieldName(selector, i));

   // For each selected field, collect its columns' pages
   for (const fieldName of selectedFields) {
      const columns = rntuple.fieldToColumns[fieldName];
      if (!columns)
         throw new Error(`Selected field '${fieldName}' not found in RNTuple`);

      for (const colDesc of columns) {
         const colEntry = builder.pageLocations[clusterIndex]?.[colDesc.index];

         // When the data is missing or broken
         if (!colEntry || !colEntry.pages)
            throw new Error(`No pages for column ${colDesc.index} in cluster ${clusterIndex}`);

         for (const page of colEntry.pages)
            pages.push({ page, colDesc, fieldName });
      }
   }

   selector.currentCluster++;

   // Early exit if no pages to read (i.e., no selected fields matched)
   if (pages.length === 0) {
      selector.Terminate(false);
      return false;
   }

   // Build flat array of [offset, size, offset, size, ...] to read pages
   const dataToRead = pages.flatMap(p =>
      [Number(p.page.locator.offset), Number(p.page.locator.size)]
   );

   return rntuple.$file.readBuffer(dataToRead).then(blobsRaw => {
      const blobs = Array.isArray(blobsRaw) ? blobsRaw : [blobsRaw],
            unzipPromises = blobs.map((blob, idx) => {
               const { page, colDesc } = pages[idx],
                     colEntry = builder.pageLocations[clusterIndex][colDesc.index], // Access column entry
                     numElements = Number(page.numElements),
                     elementSize = colDesc.bitsOnStorage / 8;

               // Check if data is compressed
               if (colEntry.compression === 0)
                  return Promise.resolve(blob); // Uncompressed: use blob directly
               const expectedSize = numElements * elementSize;

               // Special handling for boolean fields
               if (colDesc.coltype === ENTupleColumnType.kBit) {
                  const expectedBoolSize = Math.ceil(numElements / 8);
                  if (blob.byteLength === expectedBoolSize)
                     return Promise.resolve(blob);
                  // Try decompression but catch errors for boolean fields
                  return R__unzip(blob, expectedBoolSize).catch(err => {
                     throw new Error(`Failed to unzip boolean page ${idx}: ${err.message}`);
                  });
               }

               // If the blob is already the expected size, treat as uncompressed
               if (blob.byteLength === expectedSize)
                  return Promise.resolve(blob);

               // Try decompression
               return R__unzip(blob, expectedSize).then(result => {
                  if (!result)
                     return blob; // Fallback to original blob
                  return result;
               }).catch(err => {
                  throw new Error(`Failed to unzip page ${idx}: ${err.message}`);
               });
            });

      return Promise.all(unzipPromises).then(unzipBlobs => {
         rntuple._clusterData = {}; // store deserialized data per column index

         for (let i = 0; i < unzipBlobs.length; ++i) {
            const blob = unzipBlobs[i];
            // Ensure blob is a DataView
            if (!(blob instanceof DataView))
               throw new Error(`Invalid blob type for page ${i}: ${Object.prototype.toString.call(blob)}`);
            const colDesc = pages[i].colDesc,
                  values = builder.deserializePage(blob, colDesc, pages[i].page);

            // Support multiple representations (e.g., string fields with offsets + payload)
            if (!rntuple._clusterData[colDesc.index])
               rntuple._clusterData[colDesc.index] = [];

            rntuple._clusterData[colDesc.index].push(values);
         }

         const numEntries = clusterSummary.numEntries;
         for (let i = 0; i < numEntries; ++i) {
            for (let b = 0; b < selector.numBranches(); ++b) {
               const fieldName = getSelectorFieldName(selector, b),
                     tgtName = selector.nameOfBranch(b);

               selector.tgtobj[tgtName] = readEntry(rntuple, fieldName, clusterIndex, i);
            }
            selector.Process(selector.currentEntry++);
         }

         return readNextCluster(rntuple, selector);
      });
   });
}

// TODO args can later be used to filter fields, limit entries, etc.
// Create reader and deserialize doubles from the buffer
function rntupleProcess(rntuple, selector, args) {
   return readHeaderFooter(rntuple).then(() => {
      selector.Begin();
      selector.currentCluster = 0;
      selector.currentEntry = 0;
      return readNextCluster(rntuple, selector, args);
   }).then(() => selector);
}

class TDrawSelectorTuple extends TDrawSelector {

   /** @summary Return total number of entries
     * @desc TODO: check implementation details ! */
   getNumEntries(tuple) {
      let cnt = 0;
      tuple?.builder.clusterSummaries.forEach(summary => { cnt += summary.numEntries; });
      return cnt;
   }

   /** @summary Search for field in tuple
     * @desc TODO: Can be more complex when name includes extra parts referencing member or collection size or more  */
   findBranch(tuple, name) {
      return tuple.builder?.fieldDescriptors.find(field => {
         return field.fieldName === name;
      });
   }

   /** @summary Returns true if field can be used as array */
   isArrayBranch(/* tuple, br */) { return false; }

   /** @summary Begin of processing */
   Begin(tree) {
      super.Begin(tree);
      this._first_entry = true;
      this._has_bigint = false;
   }

   /** @summary Process entry */
   Process(entry) {
      if (this._first_entry || this._has_bigint) {
         for (let n = 0; n < this.numBranches(); ++n) {
            const name = this.nameOfBranch(n);
            if (typeof this.tgtobj[name] === 'bigint') {
               this.tgtobj[name] = Number(this.tgtobj[name]);
               this._has_bigint = true;
            }
         }
      }

      this._first_entry = false;

      super.Process(entry);
   }

} // class TDrawSelectorTuple


/** @summary implementation of drawing for RNTuple
 * @param {object|string} args - different setting or simply draw expression
 * @param {string} args.expr - draw expression
 * @param {string} [args.cut=undefined] - cut expression (also can be part of 'expr' after '::')
 * @param {string} [args.drawopt=undefined] - draw options for result histogram
 * @param {number} [args.firstentry=0] - first entry to process
 * @param {number} [args.numentries=undefined] - number of entries to process, all by default
 * @param {Array} [args.elist=undefined] - array of entries id to process, all by default
 * @param {boolean} [args.staged] - staged processing, first apply cut to select entries and then perform drawing for selected entries
 * @param {object} [args.branch=undefined] - TBranch object from TTree itself for the direct drawing
 * @param {function} [args.progress=undefined] - function called during histogram accumulation with obj argument
 * @return {Promise} with produced object */

async function rntupleDraw(rntuple, args) {
   if (isStr(args))
      args = { expr: args };
   else if (!isObject(args))
      args = {};

   args.SelectorClass = TDrawSelectorTuple;
   args.processFunction = rntupleProcess;

   return readHeaderFooter(rntuple).then(res_header_footer => {
      return res_header_footer ? treeDraw(rntuple, args) : null;
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
         const item = {
            _name: field.fieldName,
            _typename: 'ROOT::RNTupleField', // pseudo class name, used in draw.mjs
            _kind: 'ROOT::RNTupleField',
            _title: `Filed of type ${field.typeName}`,
            $tuple: tuple, // reference on tuple, need for drawing
            $field: field
         };

         item._obj = item;

         tuple_node._childs.push(item);
      });

      return true;
   });
}

export { tupleHierarchy, readHeaderFooter, RBufferReader, rntupleProcess, readEntry, rntupleDraw };
