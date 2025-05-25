@echo off
setlocal enabledelayedexpansion

rem HistFactory workplace setup script

for /f "tokens=*" %%g in ('root-config --etcdir') do (set ROOTETCDIR=%%g)
echo Using etcdir !ROOTETCDIR!
for /f "tokens=*" %%g in ('root-config --tutdir') do (set ROOTTUTDIR=%%g)
echo Using tutorials dir !ROOTTUTDIR!

if not "%1" == "" (
  set DIR=%~1
  echo HistFactory workplace will be created in: !DIR!
) else (
  set DIR=.
  echo HistFactory workplace will be created in the current directory
)

echo "Creating directory structure..."
mkdir !DIR!\config
mkdir !DIR!\config\examples
mkdir !DIR!\data
mkdir !DIR!\results

echo "Copying skeleton configuration files..."
copy !ROOTETCDIR!\HistFactorySchema.dtd !DIR!\config\
copy !ROOTTUTDIR!\roofit\histfactory\example.xml !DIR!\config\
copy !ROOTTUTDIR!\roofit\histfactory\example_channel.xml !DIR!\config\

copy !ROOTETCDIR!\HistFactorySchema.dtd !DIR!\config\examples
copy !ROOTTUTDIR!\roofit\histfactory\example_Expression.xml !DIR!\config\examples\
copy !ROOTTUTDIR!\roofit\histfactory\example_Expression_channel.xml !DIR!\config\examples\
copy !ROOTTUTDIR!\roofit\histfactory\example_ShapeSys.xml !DIR!\config\examples\
copy !ROOTTUTDIR!\roofit\histfactory\example_ShapeSys_channel.xml !DIR!\config\examples\
copy !ROOTTUTDIR!\roofit\histfactory\example_ShapeSys2D.xml !DIR!\config\examples\
copy !ROOTTUTDIR!\roofit\histfactory\example_ShapeSys2D_channel.xml !DIR!\config\examples\
copy !ROOTTUTDIR!\roofit\histfactory\example_DataDriven.xml !DIR!\config\examples\
copy !ROOTTUTDIR!\roofit\histfactory\example_DataDriven_signalRegion.xml !DIR!\config\examples\
copy !ROOTTUTDIR!\roofit\histfactory\example_DataDriven_controlRegion.xml !DIR!\config\examples\

rem echo "Making skeleton data files..."
root.exe -b -q !ROOTTUTDIR!\roofit\histfactory\makeExample.C
move ShapeSys.root !DIR!\data\
move ShapeSys2D.root !DIR!\data\
move StatError.root !DIR!\data\
move dataDriven.root !DIR!\data\
move example.root !DIR!\data\

echo Done!
rem echo "You can run the example with: hist2workspace !DIR!\config\example.xml"

exit /b 0

