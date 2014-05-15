
.PHONY: doxygen images clean

docdir="$(shell ./makedocdir.sh)"
imgdir="$(shell ./makeimgdir.sh)"
macdir="$(shell ./makemacrodir.sh)"

all: images doxygen

images:
	./makeallimages.sh $(imgdir) $(macdir)

doxygen:
	doxygen

clean:
	rm -r $(macdir)
	rm -r $(imgdir)
	rm -r $(docdir)
