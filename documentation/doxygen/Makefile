
.PHONY: doxygen images clean

docdir="$(shell ./makedocdir.sh)"
imgdir="$(shell ./makeimgdir.sh)"
macdir="$(shell ./makemacdir.sh)"

all: images doxygen

images:
	./makeallimages.sh $(imgdir) $(macdir)

doxygen:
	doxygen

clean:
	rm -r $(macdir)
	rm -r $(imgdir)
	rm -r $(docdir)
