#ifdef VC_GNU_ASM
template<typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T1> &x1) {
  __asm__ __volatile__(""::"x"(x1.data()));
}
template<typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T1> &x1) {
  __asm__ __volatile__("":"+x"(x1.data()));
}
template<typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T2> &x2, const Vector<T1> &x1) {
  __asm__ __volatile__(""::"x"(x2.data()), "x"(x1.data()));
}
template<typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T2> &x2, Vector<T1> &x1) {
  __asm__ __volatile__("":"+x"(x2.data()), "+x"(x1.data()));
}
template<typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T3> &x3, const Vector<T2> &x2, const Vector<T1> &x1) {
  __asm__ __volatile__(""::"x"(x3.data()), "x"(x2.data()), "x"(x1.data()));
}
template<typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T3> &x3, Vector<T2> &x2, Vector<T1> &x1) {
  __asm__ __volatile__("":"+x"(x3.data()), "+x"(x2.data()), "+x"(x1.data()));
}
template<typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T4> &x4, const Vector<T3> &x3, const Vector<T2> &x2, const Vector<T1> &x1) {
  __asm__ __volatile__(""::"x"(x4.data()), "x"(x3.data()), "x"(x2.data()), "x"(x1.data()));
}
template<typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T4> &x4, Vector<T3> &x3, Vector<T2> &x2, Vector<T1> &x1) {
  __asm__ __volatile__("":"+x"(x4.data()), "+x"(x3.data()), "+x"(x2.data()), "+x"(x1.data()));
}
template<typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T5> &x5, const Vector<T4> &x4, const Vector<T3> &x3, const Vector<T2> &x2, const Vector<T1> &x1) {
  __asm__ __volatile__(""::"x"(x5.data()), "x"(x4.data()), "x"(x3.data()), "x"(x2.data()), "x"(x1.data()));
}
template<typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T5> &x5, Vector<T4> &x4, Vector<T3> &x3, Vector<T2> &x2, Vector<T1> &x1) {
  __asm__ __volatile__("":"+x"(x5.data()), "+x"(x4.data()), "+x"(x3.data()), "+x"(x2.data()), "+x"(x1.data()));
}
template<typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T6> &x6, const Vector<T5> &x5, const Vector<T4> &x4, const Vector<T3> &x3, const Vector<T2> &x2, const Vector<T1> &x1) {
  __asm__ __volatile__(""::"x"(x6.data()), "x"(x5.data()), "x"(x4.data()), "x"(x3.data()), "x"(x2.data()), "x"(x1.data()));
}
template<typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T6> &x6, Vector<T5> &x5, Vector<T4> &x4, Vector<T3> &x3, Vector<T2> &x2, Vector<T1> &x1) {
  __asm__ __volatile__("":"+x"(x6.data()), "+x"(x5.data()), "+x"(x4.data()), "+x"(x3.data()), "+x"(x2.data()), "+x"(x1.data()));
}
template<typename T7, typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T7> &x7, const Vector<T6> &x6, const Vector<T5> &x5, const Vector<T4> &x4, const Vector<T3> &x3, const Vector<T2> &x2, const Vector<T1> &x1) {
  __asm__ __volatile__(""::"x"(x7.data()), "x"(x6.data()), "x"(x5.data()), "x"(x4.data()), "x"(x3.data()), "x"(x2.data()), "x"(x1.data()));
}
template<typename T7, typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T7> &x7, Vector<T6> &x6, Vector<T5> &x5, Vector<T4> &x4, Vector<T3> &x3, Vector<T2> &x2, Vector<T1> &x1) {
  __asm__ __volatile__("":"+x"(x7.data()), "+x"(x6.data()), "+x"(x5.data()), "+x"(x4.data()), "+x"(x3.data()), "+x"(x2.data()), "+x"(x1.data()));
}
template<typename T8, typename T7, typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T8> &x8, const Vector<T7> &x7, const Vector<T6> &x6, const Vector<T5> &x5, const Vector<T4> &x4, const Vector<T3> &x3, const Vector<T2> &x2, const Vector<T1> &x1) {
  __asm__ __volatile__(""::"x"(x8.data()), "x"(x7.data()), "x"(x6.data()), "x"(x5.data()), "x"(x4.data()), "x"(x3.data()), "x"(x2.data()), "x"(x1.data()));
}
template<typename T8, typename T7, typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T8> &x8, Vector<T7> &x7, Vector<T6> &x6, Vector<T5> &x5, Vector<T4> &x4, Vector<T3> &x3, Vector<T2> &x2, Vector<T1> &x1) {
  __asm__ __volatile__("":"+x"(x8.data()), "+x"(x7.data()), "+x"(x6.data()), "+x"(x5.data()), "+x"(x4.data()), "+x"(x3.data()), "+x"(x2.data()), "+x"(x1.data()));
}
#elif defined(VC_MSVC)
#pragma optimize("g", off)
template<typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T1> &/*x1*/) {
}
#pragma optimize("g", off)
template<typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T1> &/*x1*/) {
}
#pragma optimize("g", on)
#pragma optimize("g", off)
template<typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T2> &/*x2*/, const Vector<T1> &/*x1*/) {
}
#pragma optimize("g", off)
template<typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T2> &/*x2*/, Vector<T1> &/*x1*/) {
}
#pragma optimize("g", on)
#pragma optimize("g", off)
template<typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T3> &/*x3*/, const Vector<T2> &/*x2*/, const Vector<T1> &/*x1*/) {
}
#pragma optimize("g", off)
template<typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T3> &/*x3*/, Vector<T2> &/*x2*/, Vector<T1> &/*x1*/) {
}
#pragma optimize("g", on)
#pragma optimize("g", off)
template<typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T4> &/*x4*/, const Vector<T3> &/*x3*/, const Vector<T2> &/*x2*/, const Vector<T1> &/*x1*/) {
}
#pragma optimize("g", off)
template<typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T4> &/*x4*/, Vector<T3> &/*x3*/, Vector<T2> &/*x2*/, Vector<T1> &/*x1*/) {
}
#pragma optimize("g", on)
#pragma optimize("g", off)
template<typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T5> &/*x5*/, const Vector<T4> &/*x4*/, const Vector<T3> &/*x3*/, const Vector<T2> &/*x2*/, const Vector<T1> &/*x1*/) {
}
#pragma optimize("g", off)
template<typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T5> &/*x5*/, Vector<T4> &/*x4*/, Vector<T3> &/*x3*/, Vector<T2> &/*x2*/, Vector<T1> &/*x1*/) {
}
#pragma optimize("g", on)
#pragma optimize("g", off)
template<typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T6> &/*x6*/, const Vector<T5> &/*x5*/, const Vector<T4> &/*x4*/, const Vector<T3> &/*x3*/, const Vector<T2> &/*x2*/, const Vector<T1> &/*x1*/) {
}
#pragma optimize("g", off)
template<typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T6> &/*x6*/, Vector<T5> &/*x5*/, Vector<T4> &/*x4*/, Vector<T3> &/*x3*/, Vector<T2> &/*x2*/, Vector<T1> &/*x1*/) {
}
#pragma optimize("g", on)
#pragma optimize("g", off)
template<typename T7, typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T7> &/*x7*/, const Vector<T6> &/*x6*/, const Vector<T5> &/*x5*/, const Vector<T4> &/*x4*/, const Vector<T3> &/*x3*/, const Vector<T2> &/*x2*/, const Vector<T1> &/*x1*/) {
}
#pragma optimize("g", off)
template<typename T7, typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T7> &/*x7*/, Vector<T6> &/*x6*/, Vector<T5> &/*x5*/, Vector<T4> &/*x4*/, Vector<T3> &/*x3*/, Vector<T2> &/*x2*/, Vector<T1> &/*x1*/) {
}
#pragma optimize("g", on)
#pragma optimize("g", off)
template<typename T8, typename T7, typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T8> &/*x8*/, const Vector<T7> &/*x7*/, const Vector<T6> &/*x6*/, const Vector<T5> &/*x5*/, const Vector<T4> &/*x4*/, const Vector<T3> &/*x3*/, const Vector<T2> &/*x2*/, const Vector<T1> &/*x1*/) {
}
#pragma optimize("g", off)
template<typename T8, typename T7, typename T6, typename T5, typename T4, typename T3, typename T2, typename T1>
static Vc_ALWAYS_INLINE void forceToRegistersDirty(Vector<T8> &/*x8*/, Vector<T7> &/*x7*/, Vector<T6> &/*x6*/, Vector<T5> &/*x5*/, Vector<T4> &/*x4*/, Vector<T3> &/*x3*/, Vector<T2> &/*x2*/, Vector<T1> &/*x1*/) {
}
#pragma optimize("g", on)
#else
#error "forceToRegisters unsupported on this compiler"
#endif
