#
# Makefile for creating HTML out of Markdown for the PROOF configuration
# manual.
#
# Pandoc is used for the conversion. Thanks to Olivier Couet for the Markdown
# ROOT documentation generation scripts.
#
# Pandoc sample installation on OSX (using Homebrew):
#
#   $> brew install haskell-platform
#   $> cabal install pandoc
#   $> export PATH=$HOME/.cabal/bin:$PATH
#

PANDOC_OPTS :=-s -S --toc --chapters --number-sections -f markdown -H css/github.css

INPUT_MDS := \
	ConfigProofPoD.md \
	DeployVirtualAnalysisFacility.md

OUTPUT_HTML = $(INPUT_MDS:.md=.html)

.PHONY: all html clean

all: html

html: $(OUTPUT_HTML)

%.html: %.md
	@echo Generating HTML: $@
	@pandoc $(PANDOC_OPTS) $< -o $@

clean:
	@echo Cleaning up HTML
	@rm -f $(OUTPUT_HTML)
