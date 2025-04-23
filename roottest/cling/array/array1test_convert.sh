sed -e 's/0x[^ ]*//g' -e 's/[A-Za-z]:.*array1.C/array1.C/g' \
	-e 's/(address: NA)/ /' -e 's/, size = .*/=/' \
	-e 's/input_line_[0-9]*/array1.C/' -e 's/    4   /         3   /' -e 's/   11   /         10   /'
