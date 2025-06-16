//===-- LccrtEmitter.cpp - Lccrt mc-emittion code -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LccrtMCEmitter's classes.
//
//===----------------------------------------------------------------------===//

#include <string.h>

#include "LccrtEmitter.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/CodeGen/Lccrt.h"

#ifdef LLVM_WITH_LCCRT
using namespace llvm;

#define DEBUG_TYPE "lccrt"

void
LccrtMCCodeEmitter::encodeInstruction(const MCInst &Inst, SmallVectorImpl<char> &CB,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const
{
    llvm_unreachable( "LccrtMCCodeEmitter::encodeInstruction: TODO");

    return;
} /* LccrtMCCodeEmitter::encodeInstruction */

LccrtMCMachObjectTargetWriter::LccrtMCMachObjectTargetWriter()
  : MCMachObjectTargetWriter( 1, 0, 0)
{
    return;
} /* LccrtMCMachObjectTargetWriter */

void
LccrtMCMachObjectTargetWriter::recordRelocation( MachObjectWriter *Writer, MCAssembler &Asm, const MCAsmLayout &Layout,
                                                 const MCFragment *Fragment, const MCFixup &Fixup, MCValue Target,
                                                 uint64_t &FixedValue)
{
    llvm_unreachable( "LccrtMCMachObjectTargetWriter: TODO");

    return;
} /* LccrtMCMachObjectTargetWriter::recordRelocation */

std::unique_ptr<MCObjectWriter>
LccrtMCAsmBackend::createObjectWriter( raw_pwrite_stream &OS) const
{
    std::unique_ptr<MCELFObjectTargetWriter> MOTW = 0;
    std::unique_ptr<MCObjectWriter> r = createELFObjectWriter( std::move( MOTW), OS, /*IsLittleEndian=*/true);

    return (r);
} /* LccrtMCAsmBackend::createObjectWriter */

unsigned
LccrtMCAsmBackend::getNumFixupKinds() const
{
    llvm_unreachable( "LccrtMCAsmBackend::getNumFixupKinds: TODO");

    return (0);
} /* LccrtMCAsmBackend::getNumFixupKinds */

void
LccrtMCAsmBackend::applyFixup( const MCAssembler &Asm, const MCFixup &Fixup,
                               const MCValue &Target, MutableArrayRef<char> Data,
                               uint64_t Value, bool IsResolved, const MCSubtargetInfo *STI) const
{
    llvm_unreachable( "LccrtMCAsmBackend::applyFixup: TODO");

    return;
} /* LccrtMCAsmBackend::applyFixup */

bool
LccrtMCAsmBackend::mayNeedRelaxation( const MCInst &Inst, const MCSubtargetInfo &STI) const
{
    llvm_unreachable( "LccrtMCAsmBackend::mayNeedRelaxation: TODO");

    return (false);
} /* LccrtMCAsmBackend::mayNeedRelaxation */

bool
LccrtMCAsmBackend::fixupNeedsRelaxation( const MCFixup &Fixup, uint64_t Value, const MCRelaxableFragment *DF,
                                         const MCAsmLayout &Layout) const
{
    llvm_unreachable( "LccrtMCAsmBackend::fixupNeedsRelaxation: TODO");

    return (false);
} /* LccrtMCAsmBackend::fixupNeedsRelaxation */

void
LccrtMCAsmBackend::relaxInstruction( const MCInst &Inst, const MCSubtargetInfo &STI,
                                     MCInst &Res) const
{
    llvm_unreachable( "LccrtMCAsmBackend::relaxInstruction: TODO");

    return;
} /* LccrtMCAsmBackend::relaxInstruction */

bool
LccrtMCAsmBackend::writeNopData( raw_ostream &OS, uint64_t Count, const MCSubtargetInfo *STI) const
{
    llvm_unreachable( "LccrtMCAsmBackend::writeNopData: TODO");

    return (false);
} /* LccrtMCAsmBackend::writeNopData */

std::unique_ptr<MCObjectTargetWriter>
LccrtMCAsmBackend::createObjectTargetWriter() const
{
    llvm_unreachable( "LccrtMCAsmBackend::createObjectTargetWriter: TODO");

    return 0;
}
#endif /* LLVM_WITH_LCCRT */
