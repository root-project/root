#ifdef __ROOTCLING__

#pragma link C++ enum PadFlags;
#pragma link C++ enum PadSubset;
#pragma link C++ class CalArray<PadFlags>-;
#pragma link C++ class Event+;
#pragma link C++ class UnevenEvent+;

// With v6.34 old:
// StreamerInfo for class: Event, checksum=0xf5c1301b
// StreamerInfo for class: UnevenEvent, checksum=0xd5c80dc9

// #pragma read sourceClass="Event" checksums="[0xa2558fd6]" targetClass="Event" source="std::vector<int> mFlags" target="mFlags" code="{ mFlags.clear(); auto data = reinterpret_cast<unsigned short*>(onfile.mFlags.data()); constexpr size_t delta = sizeof(int)/sizeof(unsigned short); size_t nvalues = onfile.mFlags.size() / delta; for(size_t i = 0; i < nvalues; ++i) mFlags.push_back(static_cast<PadFlags>( data[i] )); }"

#pragma read sourceClass="Event" checksum="[0xf5c1301b]" targetClass="Event" \
   source="std::vector<int> mFlags" target="mFlags" \
   code="{ LoadEnumCollection(onfile.mFlags, mFlags); }"

//#pragma read sourceClass="CalArray<PadFlags>" checksums="[0xc75fb860]" targetClass="CalArray<PadFlags>" source="std::vector<PadFlags> mFlags" target="mFlags" code="{ LoadEnumCollection(onfile.mFlags, mFlags); }"

#pragma read sourceClass="UnevenEvent" checksum="[0xd5c80dc9]" targetClass="UnevenEvent" \
    source="std::vector<int> mPad" target="mPad" \
    code="{ LoadEnumCollection(onfile.mPad, mPad); }"

#pragma read sourceClass="UnevenEvent" checksum="[0xd5c80dc9]" targetClass="UnevenEvent" \
    source="std::vector<int> mChar" target="mChar" \
    code="{ LoadEnumCollection(onfile.mChar, mChar); }"


#pragma link C++ class CalArray2<PadFlags>+;

#pragma read sourceClass="CalArray<PadFlags>" targetClass="CalArray2<PadFlags>";

#endif
