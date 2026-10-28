/** © 2016 Nathan Rugg <nmrugg@gmail.com>
  * Code extracted from https://github.com/LZMA-JS/LZMA-JS */

const __4294967296 = 4294967296,
      P0_longLit = [0, 0],
      P1_longLit = [1, 0];

function initDim(len) {
   // This is MUCH faster than "new Array(len)" in newer versions of v8 (starting with Node.js 0.11.15, which uses v8 3.28.73).
   const a = [];
   a[len - 1] = undefined;
   return a;
}

function add(a, b) {
   return create(a[0] + b[0], a[1] + b[1]);
}

function compare(a, b) {
   if (a[0] === b[0] && a[1] === b[1])
      return 0;
   const nega = a[1] < 0,
         negb = b[1] < 0;
   if (nega && !negb)
      return -1;
   if (!nega && negb)
      return 1;
   if (sub(a, b)[1] < 0)
      return -1;
   return 1;
}

function create(valueLow, valueHigh) {
   valueHigh %= 1.8446744073709552E19;
   valueLow %= 1.8446744073709552E19;
   const diffHigh = valueHigh % __4294967296,
      diffLow = Math.floor(valueLow / __4294967296) * __4294967296;
   valueHigh = valueHigh - diffHigh + diffLow;
   valueLow = valueLow - diffLow + diffHigh;
   while (valueLow < 0) {
      valueLow += __4294967296;
      valueHigh -= __4294967296;
   }
   while (valueLow > 4294967295) {
      valueLow -= __4294967296;
      valueHigh += __4294967296;
   }
   valueHigh = valueHigh % 1.8446744073709552E19;
   while (valueHigh > 9223372032559808512)
      valueHigh -= 1.8446744073709552E19;
   while (valueHigh < -9223372036854775808)
      valueHigh += 1.8446744073709552E19;
   return [valueLow, valueHigh];
}

function fromInt(value) {
   if (value >= 0)
      return [value, 0];
   else
      return [value + __4294967296, -__4294967296];
}

function lowBits_0(a) {
   if (a[0] >= 2147483648)
      return ~~Math.max(Math.min(a[0] - __4294967296, 2147483647), -2147483648);
   else
      return ~~Math.max(Math.min(a[0], 2147483647), -2147483648);
}

function sub(a, b) {
   return create(a[0] - b[0], a[1] - b[1]);
}

function $ByteArrayInputStream(this$static, buf, offset) {
   this$static.buf = buf;
   this$static.pos = offset ?? 0;
   this$static.count = buf.length;
   return this$static;
}

function $read(this$static) {
   if (this$static.pos >= this$static.count)
      return -1;
   return this$static.buf[this$static.pos++]; //  & 255; not needed, input always uint8
}

function $ByteArrayOutputStream(this$static, buf) {
   this$static.buf = buf;
   this$static.count = 0;
   return this$static;
}

function $write_0(this$static, buf, off, len) {
   arraycopy(buf, off, this$static.buf, this$static.count, len);
   this$static.count += len;
}

function arraycopy(src, srcOfs, dest, destOfs, len) {
   dest.set(src.subarray(srcOfs, srcOfs + len), destOfs);
}

function $CopyBlock(this$static, distance, len) {
   let pos = this$static._pos - distance - 1;
   if (pos < 0)
      pos += this$static._windowSize;

   for (; len; --len) {
      if (pos >= this$static._windowSize)
         pos = 0;

      this$static._buffer[this$static._pos++] = this$static._buffer[pos++];
      if (this$static._pos >= this$static._windowSize)
         $Flush_0(this$static);
   }
}

function $Create_5(this$static, windowSize) {
   if (!this$static._buffer || this$static._windowSize !== windowSize)
      this$static._buffer = new Uint8Array(windowSize);

   this$static._windowSize = windowSize;
   this$static._pos = 0;
   this$static._streamPos = 0;
}

function $Flush_0(this$static) {
   const size = this$static._pos - this$static._streamPos;
   if (!size)
      return;

   $write_0(this$static._stream, this$static._buffer, this$static._streamPos, size);
   if (this$static._pos >= this$static._windowSize)
      this$static._pos = 0;

   this$static._streamPos = this$static._pos;
}

function $GetByte(this$static, distance) {
   let pos = this$static._pos - distance - 1;
   if (pos < 0)
      pos += this$static._windowSize;

   return this$static._buffer[pos];
}

function $PutByte(this$static, b) {
   this$static._buffer[this$static._pos++] = b;
   if (this$static._pos >= this$static._windowSize)
      $Flush_0(this$static);
}

function $ReleaseStream(this$static) {
   $Flush_0(this$static);
   this$static._stream = null;
}

function GetLenToPosState(len) {
   len -= 2;
   return (len < 4) ? len : 3;
}

function StateUpdateChar(index) {
   if (index < 4)
      return 0;

   return index < 10 ? index - 3 : index - 6;
}

/** @summary Decode LZMA2 chunks, returns size of unpacked data */
function $DecodeLZMA2(this$static, input, output_size) {
   const rc = this$static.m_RangeDecoder,
         window = this$static.m_OutWindow;
   let produced = 0, needProps = true, needDictReset = true, control;

   rc.Stream = input;
   this$static.state = 0;
   this$static.rep0 = 0;
   this$static.rep1 = 0;
   this$static.rep2 = 0;
   this$static.rep3 = 0;
   this$static.nowPos64 = P0_longLit;
   this$static.prevByte = 0;

   while ((control = $read(input)) !== 0) {
      if (control < 0)
         throw new Error('LZMA: unexpected end of input');

      if (control === 1 || control >= 0xE0) {
         needDictReset = false;
         needProps = true;
         this$static.nowPos64 = P0_longLit;
         this$static.prevByte = 0;
      } else if (needDictReset)
         throw new Error('corrupted input');

      let size, packed, prevByte;

      if (control < 0x80) {
         if (control > 2)
            throw new Error('corrupted input');
         size = ($read(input) << 8 | $read(input)) + 1;
         if (produced + size > output_size)
            throw new Error('corrupted input');
         for (let i = 0; i < size; ++i) {
            prevByte = $read(input);
            if (prevByte < 0)
               throw new Error('LZMA: unexpected end of input');
            $PutByte(window, prevByte);
         }
         this$static.prevByte = prevByte;
         this$static.nowPos64 = add(this$static.nowPos64, fromInt(size));
         produced += size;
         continue;
      }

      size = ((control & 0x1F) << 16 | $read(input) << 8 | $read(input)) + 1;
      packed = ($read(input) << 8 | $read(input)) + 1;

      const mode = control >> 5 & 3;
      if (mode >= 2) {
         const val = $read(input);
         if (val < 0 || val >= 9 * 5 * 5)
            throw new Error('corrupted input');
         const lc = val % 9, lp = ~~(val / 9) % 5, pb = ~~(val / 45);
         if (lc + lp > 4 || !$SetLcLpPb(this$static, lc, lp, pb))
            throw new Error('corrupted input');
         needProps = false;
      } else if (needProps)
         throw new Error('corrupted input');

      if (produced + size > output_size || input.pos + packed > input.count)
         throw new Error('corrupted input');

      if (mode >= 1) {
         $Init_1(this$static);
         this$static.state = 0;
         this$static.rep0 = 0;
         this$static.rep1 = 0;
         this$static.rep2 = 0;
         this$static.rep3 = 0;
      }

      const chunkEnd = input.pos + packed;
      this$static.outSize = add(this$static.nowPos64, fromInt(size));
      $Init_8(rc);

      while (compare(this$static.nowPos64, this$static.outSize) < 0) {
         if ($CodeOneChunk(this$static))
            throw new Error('corrupted input');
      }

      if (compare(this$static.nowPos64, this$static.outSize) || input.pos !== chunkEnd)
         throw new Error('corrupted input');

      produced += size;
   }

   $ReleaseStream(window);
   return produced;
}

function $CodeOneChunk(this$static) {
   let decoder2, distance, len, numDirectBits, posSlot;
   const posState = lowBits_0(this$static.nowPos64) & this$static.m_PosStateMask;
   if (!$DecodeBit(this$static.m_RangeDecoder, this$static.m_IsMatchDecoders, (this$static.state << 4) + posState)) {
      decoder2 = $GetDecoder(this$static.m_LiteralDecoder, lowBits_0(this$static.nowPos64), this$static.prevByte);
      if (this$static.state < 7)
         this$static.prevByte = $DecodeNormal(decoder2, this$static.m_RangeDecoder);
      else
         this$static.prevByte = $DecodeWithMatchByte(decoder2, this$static.m_RangeDecoder, $GetByte(this$static.m_OutWindow, this$static.rep0));

      $PutByte(this$static.m_OutWindow, this$static.prevByte);
      this$static.state = StateUpdateChar(this$static.state);
      this$static.nowPos64 = add(this$static.nowPos64, P1_longLit);
   } else {
      if ($DecodeBit(this$static.m_RangeDecoder, this$static.m_IsRepDecoders, this$static.state)) {
         len = 0;
         if (!$DecodeBit(this$static.m_RangeDecoder, this$static.m_IsRepG0Decoders, this$static.state)) {
            if (!$DecodeBit(this$static.m_RangeDecoder, this$static.m_IsRep0LongDecoders, (this$static.state << 4) + posState)) {
               this$static.state = this$static.state < 7 ? 9 : 11;
               len = 1;
            }
         } else {
            if (!$DecodeBit(this$static.m_RangeDecoder, this$static.m_IsRepG1Decoders, this$static.state))
               distance = this$static.rep1;
            else {
               if (!$DecodeBit(this$static.m_RangeDecoder, this$static.m_IsRepG2Decoders, this$static.state))
                  distance = this$static.rep2;
               else {
                  distance = this$static.rep3;
                  this$static.rep3 = this$static.rep2;
               }
               this$static.rep2 = this$static.rep1;
            }
            this$static.rep1 = this$static.rep0;
            this$static.rep0 = distance;
         }
         if (!len) {
            len = $Decode(this$static.m_RepLenDecoder, this$static.m_RangeDecoder, posState) + 2;
            this$static.state = this$static.state < 7 ? 8 : 11;
         }
      } else {
         this$static.rep3 = this$static.rep2;
         this$static.rep2 = this$static.rep1;
         this$static.rep1 = this$static.rep0;
         len = 2 + $Decode(this$static.m_LenDecoder, this$static.m_RangeDecoder, posState);
         this$static.state = this$static.state < 7 ? 7 : 10;
         posSlot = $Decode_0(this$static.m_PosSlotDecoder[GetLenToPosState(len)], this$static.m_RangeDecoder);
         if (posSlot >= 4) {
            numDirectBits = (posSlot >> 1) - 1;
            this$static.rep0 = (2 | posSlot & 1) << numDirectBits;
            if (posSlot < 14)
               this$static.rep0 += ReverseDecode(this$static.m_PosDecoders, this$static.rep0 - posSlot - 1, this$static.m_RangeDecoder, numDirectBits);
            else {
               this$static.rep0 += $DecodeDirectBits(this$static.m_RangeDecoder, numDirectBits - 4) << 4;
               this$static.rep0 += $ReverseDecode(this$static.m_PosAlignDecoder, this$static.m_RangeDecoder);
               if (this$static.rep0 < 0) {
                  if (this$static.rep0 === -1)
                     return 1;

                  return -1;
               }
            }
         } else
            this$static.rep0 = posSlot;
      }
      if (compare(fromInt(this$static.rep0), this$static.nowPos64) >= 0 || this$static.rep0 >= this$static.m_DictionarySizeCheck)
         return -1;

      $CopyBlock(this$static.m_OutWindow, this$static.rep0, len);
      this$static.nowPos64 = add(this$static.nowPos64, fromInt(len));
      this$static.prevByte = $GetByte(this$static.m_OutWindow, 0);
   }
   return 0;
}

function $Decoder(this$static) {
   this$static.m_OutWindow = {};
   this$static.m_RangeDecoder = {};
   this$static.m_IsMatchDecoders = initDim(192);
   this$static.m_IsRepDecoders = initDim(12);
   this$static.m_IsRepG0Decoders = initDim(12);
   this$static.m_IsRepG1Decoders = initDim(12);
   this$static.m_IsRepG2Decoders = initDim(12);
   this$static.m_IsRep0LongDecoders = initDim(192);
   this$static.m_PosSlotDecoder = initDim(4);
   this$static.m_PosDecoders = initDim(114);
   this$static.m_PosAlignDecoder = $BitTreeDecoder({}, 4);
   this$static.m_LenDecoder = $Decoder$LenDecoder({});
   this$static.m_RepLenDecoder = $Decoder$LenDecoder({});
   this$static.m_LiteralDecoder = {};
   for (let i = 0; i < 4; ++i)
      this$static.m_PosSlotDecoder[i] = $BitTreeDecoder({}, 6);

   return this$static;
}

function $Init_1(this$static) {
   InitBitModels(this$static.m_IsMatchDecoders);
   InitBitModels(this$static.m_IsRep0LongDecoders);
   InitBitModels(this$static.m_IsRepDecoders);
   InitBitModels(this$static.m_IsRepG0Decoders);
   InitBitModels(this$static.m_IsRepG1Decoders);
   InitBitModels(this$static.m_IsRepG2Decoders);
   InitBitModels(this$static.m_PosDecoders);
   $Init_0(this$static.m_LiteralDecoder);
   for (let i = 0; i < 4; ++i)
      InitBitModels(this$static.m_PosSlotDecoder[i].Models);

   $Init(this$static.m_LenDecoder);
   $Init(this$static.m_RepLenDecoder);
   InitBitModels(this$static.m_PosAlignDecoder.Models);
}

function $SetDictionarySize(this$static, dictionarySize, outputSize) {
   this$static.m_DictionarySizeCheck = Math.max(dictionarySize, 1);
   $Create_5(this$static.m_OutWindow, Math.max(Math.min(this$static.m_DictionarySizeCheck, outputSize), 4096));
}

function $SetLcLpPb(this$static, lc, lp, pb) {
   if (lc > 8 || lp > 4 || pb > 4)
      return 0;

   $Create_0(this$static.m_LiteralDecoder, lp, lc);
   const numPosStates = 1 << pb;
   $Create(this$static.m_LenDecoder, numPosStates);
   $Create(this$static.m_RepLenDecoder, numPosStates);
   this$static.m_PosStateMask = numPosStates - 1;
   return 1;
}

function $Create(this$static, numPosStates) {
   for (; this$static.m_NumPosStates < numPosStates; ++this$static.m_NumPosStates) {
      this$static.m_LowCoder[this$static.m_NumPosStates] = $BitTreeDecoder({}, 3);
      this$static.m_MidCoder[this$static.m_NumPosStates] = $BitTreeDecoder({}, 3);
   }
}

function $Decode(this$static, rangeDecoder, posState) {
   if (!$DecodeBit(rangeDecoder, this$static.m_Choice, 0))
      return $Decode_0(this$static.m_LowCoder[posState], rangeDecoder);

   let symbol = 8;
   if (!$DecodeBit(rangeDecoder, this$static.m_Choice, 1))
      symbol += $Decode_0(this$static.m_MidCoder[posState], rangeDecoder);
   else
      symbol += 8 + $Decode_0(this$static.m_HighCoder, rangeDecoder);

   return symbol;
}

function $Decoder$LenDecoder(this$static) {
   this$static.m_Choice = initDim(2);
   this$static.m_LowCoder = initDim(16);
   this$static.m_MidCoder = initDim(16);
   this$static.m_HighCoder = $BitTreeDecoder({}, 8);
   this$static.m_NumPosStates = 0;
   return this$static;
}

function $Init(this$static) {
   InitBitModels(this$static.m_Choice);
   for (let posState = 0; posState < this$static.m_NumPosStates; ++posState) {
      InitBitModels(this$static.m_LowCoder[posState].Models);
      InitBitModels(this$static.m_MidCoder[posState].Models);
   }
   InitBitModels(this$static.m_HighCoder.Models);
}


function $Create_0(this$static, numPosBits, numPrevBits) {
   if (this$static.m_Coders != null && this$static.m_NumPrevBits === numPrevBits && this$static.m_NumPosBits === numPosBits)
      return;
   this$static.m_NumPosBits = numPosBits;
   this$static.m_PosMask = (1 << numPosBits) - 1;
   this$static.m_NumPrevBits = numPrevBits;
   const numStates = 1 << this$static.m_NumPrevBits + this$static.m_NumPosBits;
   this$static.m_Coders = initDim(numStates);
   for (let i = 0; i < numStates; ++i)
      this$static.m_Coders[i] = $Decoder$LiteralDecoder$Decoder2({});
}

function $GetDecoder(this$static, pos, prevByte) {
   return this$static.m_Coders[((pos & this$static.m_PosMask) << this$static.m_NumPrevBits) + ((prevByte & 255) >>> 8 - this$static.m_NumPrevBits)];
}

function $Init_0(this$static) {
   const numStates = 1 << this$static.m_NumPrevBits + this$static.m_NumPosBits;
   for (let i = 0; i < numStates; ++i)
      InitBitModels(this$static.m_Coders[i].m_Decoders);
}

function $DecodeNormal(this$static, rangeDecoder) {
   let symbol = 1;
   do
      symbol = symbol << 1 | $DecodeBit(rangeDecoder, this$static.m_Decoders, symbol);
   while (symbol < 256);
   return symbol << 24 >> 24;
}

function $DecodeWithMatchByte(this$static, rangeDecoder, matchByte) {
   let bit, matchBit, symbol = 1;
   do {
      matchBit = matchByte >> 7 & 1;
      matchByte <<= 1;
      bit = $DecodeBit(rangeDecoder, this$static.m_Decoders, (1 + matchBit << 8) + symbol);
      symbol = symbol << 1 | bit;
      if (matchBit !== bit) {
         while (symbol < 256)
            symbol = symbol << 1 | $DecodeBit(rangeDecoder, this$static.m_Decoders, symbol);

         break;
      }
   } while (symbol < 256);
   return symbol << 24 >> 24;
}

function $Decoder$LiteralDecoder$Decoder2(this$static) {
   this$static.m_Decoders = initDim(768);
   return this$static;
}

function $BitTreeDecoder(this$static, numBitLevels) {
   this$static.NumBitLevels = numBitLevels;
   this$static.Models = initDim(1 << numBitLevels);
   return this$static;
}

function $Decode_0(this$static, rangeDecoder) {
   let m = 1;
   for (let bitIndex = this$static.NumBitLevels; bitIndex; --bitIndex)
      m = (m << 1) + $DecodeBit(rangeDecoder, this$static.Models, m);

   return m - (1 << this$static.NumBitLevels);
}

function $ReverseDecode(this$static, rangeDecoder) {
   let bit, bitIndex, m = 1, symbol = 0;
   for (bitIndex = 0; bitIndex < this$static.NumBitLevels; ++bitIndex) {
      bit = $DecodeBit(rangeDecoder, this$static.Models, m);
      m <<= 1;
      m += bit;
      symbol |= bit << bitIndex;
   }
   return symbol;
}

function ReverseDecode(Models, startIndex, rangeDecoder, NumBitLevels) {
   let bit, bitIndex, m = 1, symbol = 0;
   for (bitIndex = 0; bitIndex < NumBitLevels; ++bitIndex) {
      bit = $DecodeBit(rangeDecoder, Models, startIndex + m);
      m <<= 1;
      m += bit;
      symbol |= bit << bitIndex;
   }
   return symbol;
}

function $DecodeBit(this$static, probs, index) {
   const prob = probs[index],
         newBound = (this$static.Range >>> 11) * prob;
   if ((this$static.Code ^ -2147483648) < (newBound ^ -2147483648)) {
      this$static.Range = newBound;
      probs[index] = prob + (2048 - prob >>> 5) << 16 >> 16;
      if (!(this$static.Range & -16777216)) {
         this$static.Code = this$static.Code << 8 | $read(this$static.Stream);
         this$static.Range <<= 8;
      }
      return 0;
   } else {
      this$static.Range -= newBound;
      this$static.Code -= newBound;
      probs[index] = prob - (prob >>> 5) << 16 >> 16;
      if (!(this$static.Range & -16777216)) {
         this$static.Code = this$static.Code << 8 | $read(this$static.Stream);
         this$static.Range <<= 8;
      }
      return 1;
   }
}

function $DecodeDirectBits(this$static, numTotalBits) {
   let i, t, result = 0;
   for (i = numTotalBits; i; --i) {
      this$static.Range >>>= 1;
      t = this$static.Code - this$static.Range >>> 31;
      this$static.Code -= this$static.Range & t - 1;
      result = result << 1 | 1 - t;
      if (!(this$static.Range & -16777216)) {
         this$static.Code = this$static.Code << 8 | $read(this$static.Stream);
         this$static.Range <<= 8;
      }
   }
   return result;
}

function $Init_8(this$static) {
   this$static.Code = 0;
   this$static.Range = -1;
   for (let i = 0; i < 5; ++i)
      this$static.Code = this$static.Code << 8 | $read(this$static.Stream);
}

function InitBitModels(probs) {
   for (let i = probs.length - 1; i >= 0; --i)
      probs[i] = 1024;
}

/** @summary decompress ROOT LZMA buffer
  * @desc ROOT buffer is XZ stream with single block of LZMA2 data */
function decompress(uint8arr, tgt8arr, expected_size) {
   let pos = 14;

   const readVarInt = () => {
      let res = 0, mult = 1, b;
      do {
         b = uint8arr[pos++];
         res += (b & 0x7F) * mult;
         mult *= 128;
      } while ((b & 0x80) && (mult < 2 ** 49));
      return res;
   };

   if ((uint8arr.length < 32) || [0xFD, 0x37, 0x7A, 0x58, 0x5A, 0].some((v, i) => uint8arr[i] !== v))
      throw new Error('LZMA: wrong XZ stream header');

   const hdrend = 12 + (uint8arr[12] + 1) * 4, flags = uint8arr[13];

   if (flags & 0x40) readVarInt();
   if (flags & 0x80) readVarInt();

   // only single LZMA2 filter is supported
   if ((flags & 3) || (readVarInt() !== 0x21) || (readVarInt() !== 1) || (hdrend > uint8arr.length) || (pos >= hdrend))
      throw new Error('LZMA: unsupported XZ block');

   const dictprop = uint8arr[pos];
   if (dictprop > 40)
      throw new Error('LZMA: wrong dictionary size');
   const dictsize = dictprop === 40 ? 0xFFFFFFFF : (2 | (dictprop & 1)) * 2 ** ((dictprop >> 1) + 11),
         decoder = $Decoder({}),
         output = $ByteArrayOutputStream({}, tgt8arr);

   $SetDictionarySize(decoder, dictsize, expected_size);
   decoder.m_OutWindow._stream = output;

   const res_length = $DecodeLZMA2(decoder, $ByteArrayInputStream({}, uint8arr, hdrend), expected_size);

   if (res_length !== expected_size || output.count !== expected_size)
      throw Error(`LZMA: mismatch unpacked buffer size ${res_length} != ${expected_size}`);

   return res_length;
}

export { decompress };
