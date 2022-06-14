## \file
## \ingroup tutorial_pyroot
## Example frame with one box where the user can increase or decrease a number
## and the shown value will be updated accordingly.
##
## \macro_code
##
## \author Wim Lavrijsen

import ROOT


class pMyMainFrame(ROOT.TGMainFrame):
    def __init__(self, parent, width, height):
        ROOT.TGMainFrame.__init__(self, parent, width, height)

        self.fHor1 = ROOT.TGHorizontalFrame(self, 60, 20, ROOT.kFixedWidth)
        self.fExit = ROOT.TGTextButton(self.fHor1, "&Exit", "gApplication->Terminate(0)")
        self.fExit.SetCommand('TPython::Exec( "raise SystemExit" )')
        self.fHor1.AddFrame(self.fExit, ROOT.TGLayoutHints(
            ROOT.kLHintsTop | ROOT.kLHintsLeft | ROOT.kLHintsExpandX, 4, 4, 4, 4))
        self.AddFrame(self.fHor1, ROOT.TGLayoutHints(ROOT.kLHintsBottom | ROOT.kLHintsRight, 2, 2, 5, 1))

        self.fNumber = ROOT.TGNumberEntry(self, 0, 9, 999, ROOT.TGNumberFormat.kNESInteger,
                                          ROOT.TGNumberFormat.kNEANonNegative,
                                          ROOT.TGNumberFormat.kNELLimitMinMax,
                                          0, 99999)
        self.fLabelDispatch = ROOT.TPyDispatcher(self.DoSetlabel)
        self.fNumber.Connect("ValueSet(Long_t)", "TPyDispatcher", self.fLabelDispatch, "Dispatch()")
        self.fNumber.GetNumberEntry().Connect("ReturnPressed()", "TPyDispatcher", self.fLabelDispatch, "Dispatch()")
        self.AddFrame(self.fNumber, ROOT.TGLayoutHints(ROOT.kLHintsTop | ROOT.kLHintsLeft, 5, 5, 5, 5))
        self.fGframe = ROOT.TGGroupFrame(self, "Value")
        self.fLabel = ROOT.TGLabel(self.fGframe, "No input.")
        self.fGframe.AddFrame(self.fLabel, ROOT.TGLayoutHints(ROOT.kLHintsTop | ROOT.kLHintsLeft, 5, 5, 5, 5))
        self.AddFrame(self.fGframe, ROOT.TGLayoutHints(ROOT.kLHintsExpandX, 2, 2, 1, 1))

        self.SetCleanup(ROOT.kDeepCleanup)
        self.SetWindowName("Number Entry")
        self.MapSubwindows()
        self.Resize(self.GetDefaultSize())
        self.MapWindow()

    def __del__(self):
        self.Cleanup()

    def DoSetlabel(self):
        self.fLabel.SetText(ROOT.Form("%d" % self.fNumber.GetNumberEntry().GetIntNumber()))
        self.fGframe.Layout()


if __name__ == "__main__":
    window = pMyMainFrame(ROOT.gClient.GetRoot(), 50, 50)
