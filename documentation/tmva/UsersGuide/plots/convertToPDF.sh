
# migration tool for pdflatex
# please cd to plots directory before executing

for file in `ls *.eps`; do
    echo converting file $file
    epstopdf $file
done

