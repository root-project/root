namespace o2::dataformats {

template <int NBIdx, int NBSrc, int NBFlg>
class AbstractRef {
   template <int NBIT>
   static constexpr auto MVAR()
   {
      typename std::conditional<
         (NBIT > 32), uint64_t,
         typename std::conditional<(NBIT > 16), uint32_t,
                                   typename std::conditional<(NBIT > 8), uint16_t, uint8_t>::type>::type>::type tp = 0;
      return tp;
   }

public:
   using Base_t                     = decltype(AbstractRef::MVAR<NBIdx + NBSrc + NBFlg>());
   using Idx_t                      = decltype(AbstractRef::MVAR<NBIdx>());
   using Src_t                      = decltype(AbstractRef::MVAR<NBSrc>());
   static constexpr Base_t BaseMask = Base_t((((0x1U << (NBIdx + NBSrc + NBFlg - 1)) - 1) << 1) + 1);
   static constexpr Idx_t  IdxMask  = Idx_t((((0x1U << (NBIdx - 1)) - 1) << 1) + 1);
   static constexpr Src_t  SrcMask  = Src_t((((0x1U << (NBSrc - 1)) - 1) << 1) + 1);

protected:
   Base_t mRef = IdxMask | (SrcMask << NBIdx); // packed reference, dummy by default
};

} // namespace o2::dataformats

#ifdef __ROOTCLING__
#pragma link C++ class o2::dataformats::AbstractRef<25,5,2>+;
#endif

