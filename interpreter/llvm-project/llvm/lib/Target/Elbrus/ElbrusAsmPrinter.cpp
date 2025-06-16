#include "TargetInfo/ElbrusTargetInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCInstPrinter.h"

using namespace llvm;

class ElbrusAsmPrinter : public AsmPrinter
{
  public:
    explicit ElbrusAsmPrinter( TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
        : AsmPrinter( TM, std::move( Streamer)) {}

#if 0
    const char *getPassName() const override
    {
        return "Elbrus Assembly Printer";
    }

    void getAnalysisUsage( AnalysisUsage &AU) const override
    {
        //AU.setPreservesAll();
        //MachineFunctionPass::getAnalysisUsage( AU);
        AU.addRequired<MachineFunctionAnalysis>();
        //AU.addPreserved<MachineFunctionAnalysis>();
        //AU.addRequired<MachineModuleInfo>();
        //AU.addRequired<GCModuleInfo>();
    }
#endif

    bool runOnMachineFunction( MachineFunction &MF) override
    {
        return (false);
        //return AsmPrinter::runOnMachineFunction( MF);
    }
};

static AsmPrinter *
createElbrusAsmPrinterPass( TargetMachine &TM, std::unique_ptr<MCStreamer> &&Streamer)
{
    AsmPrinter *r = new ElbrusAsmPrinter( TM, std::move( Streamer));

    return (r);
} /* createElbrusAsmPrinterPass */

extern "C" LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeElbrusAsmPrinter()
{
    TargetRegistry::RegisterAsmPrinter( getTheElbrus32Target(),  createElbrusAsmPrinterPass);
    TargetRegistry::RegisterAsmPrinter( getTheElbrus64Target(),  createElbrusAsmPrinterPass);
    TargetRegistry::RegisterAsmPrinter( getTheElbrus128Target(), createElbrusAsmPrinterPass);

    return;
}

