# Reference SVG files for stressGraphics tests

Used for `test-stressgraphics-svg` test, to run:

    ctest -R test-stressgraphics-svg


If test failing while code is changed and reference files need to be updated,
run stressGraphics with following arguments:

    ./stressGraphics -k --web=off --svg=/home/user/git/root/test/svg_ref/

`-k` in the arguments list allow to overwrite reference files so one can directly
create PR with updated files.

But before updating reference file - please check changed images.
Maybe produce graphics is wrong and one better need to fix code which produces it.