#include "TestSharedLib.h"

int ret_zero() { return 0; }

OverlayBase::OverlayBase() {}
OverlayBase::~OverlayBase() {}
int OverlayBase::frob(int x) { return x; }

int OverlayDispatchOnce(OverlayBase* b, int x) { return b->frob(x); }

void* singleton_fixture_meyers_addr() { return &SingletonFixture::get(); }
int singleton_fixture_meyers_value() { return SingletonFixture::get().value; }
void* singleton_fixture_member_addr() {
  return &SingletonFixture::s_inline_member;
}
int singleton_fixture_member_value() {
  return SingletonFixture::s_inline_member;
}

DynamicInitSingleton::DynamicInitSingleton() : value(7) {}
void* dynamic_singleton_addr() { return &DynamicInitSingleton::get(); }
int dynamic_singleton_value() { return DynamicInitSingleton::get().value; }

// Mutable on purpose: the JIT-side copy is the demotion target under test.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int g_used_singleton = 0;
void* g_used_singleton_addr() { return &g_used_singleton; }
int g_used_singleton_value() { return g_used_singleton; }
