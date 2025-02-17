\addtogroup tutorial_tree

@{
[TTree](classTTree.html) represents a columnar dataset. Used for example by all LHC (Large Hadron Collider) experiments. Trees are optimized for reduced disk space and selecting, high-throughput columnar access with reduced memory usage.

...

In a nutshell:
~~~{.cpp}
auto file = TFile::Open("myfile.root", "recreate");
auto tree = new TTree("mytree", "A tree with my data");
...
~~~

Explore the examples below or go to [TTree class reference](classTTree.html).

## Tutorials sorted after groups
- [Basic Usage](\ref basic)
- [Copying Trees](\ref copy)
- [Collection Classes](\ref tree_collections)
- [Advanced examples](\ref advanced)
- [Graphics](\ref tree_graphics)
- [More tutorials](\ref tree_other)


[List of all tutorials](\ref tree_alltutorials)
\anchor basic
## Basic usage

These examples show how data can be stored to and read back from TTree.

| **Tutorial** | **Description** |
|------|-----------------|
| tree101_basic.C | Basic TTree usage. |
| tree102_basic.C | A variant of tree101_basic.C |
| tree103_tree.C | Simple Event class example |
| tree104_tree.C | A variant of hsimple.C but using a [TTree](classTTree.html) instead of a [TNtuple](classTNtuple.html) |
| tree105_tree.C | Illustrates how to make a tree from variables or arrays in a C struct - without a dictionary, by creating the branches for builtin types (int, float, double) and arrays explicitly |
| tree106_tree.C | The same as `tree105_tree.C`, but uses a class instead of a C-struct |
| tree107_tree.C | Example of a tree where branches are variable length arrays |
| tree108_tree.C | This example writes a tree with objects of the class `Event` |
| tree109_friend.C | Illustrates how to use tree friends |
| tree113_getval.C | Illustrates how to retrieve tree variables in arrays |
| tree114_circular.C | Example of a circular Tree. Circular Trees are interesting in online real time environments to store the results of the last maxEntries events |

\anchor copy
## Copying Trees

These examples shows how to copy the content of trees.

| **Tutorial** | **Description** |
|------|-----------------|
| tree110_copy.C | Copy a subset of a tree to a new tree |
| tree111_copy.C | Copy a subset of a tree to a new tree, one branch in a separate file |
| tree112_copy.C | Copy a subset of a tree to a new tree, selecting entries |


\anchor tree_collections
## Collection Classes

These examples show how to write and read several collection classes (std::vector, [TClonesArray](classTClonesArray.html),...) to trees.

| **Tutorial** | **Description** |
|------|-----------------|
| tree120_ntuple.C | Read a [TNtuple](classTNtuple.html) from a file and performs a simple analysis |
| tree121_hvector.C | Write and read STL vectors in a tree |
| tree122_vector3.C | Write and read a `Vector3` class in a tree|
| tree123_clonesarray.C | Write and read a [TClonesArray](classTClonesArray.html) to a tree |


\anchor advanced
## More advanced examples

These examples shows a couple of more advanced examples using complex data.

| **Tutorial** | **Description** |
|------|-----------------|
| tree130_jets.C | Usage of a tree using the `JetEvent` class. The `JetEvent` class has several collections ([TClonesArray](classTClonesArray.html)) and other collections ([TRefArray](classTRefArray.html)) referencing objects in the [TClonesArrays](classTClonesArray.html) |
| tree131_clones_event.C | Example to write & read a Tree built with a complex class inheritance tree. It demonstrates usage of inheritance and TClonesArrays |


\anchor tree_graphics
## Graphics

These tutorials show how to generate complex graphics from [TNtuple](classTNtuple.html) and/or [TTree](classTTree.html).

| **Tutorial** | **Description** |
|------|-----------------|
| tree140_spider.C | Example of a [TSpider](classTSpider.html) plot generated from a [TNtuple](classTNtuple.html) (from hsimple.root) |
| tree141_parallelcoord.C | Illustrate the use of the [TParallelCoord](classTParallelCoord.html) class (from a [TNtuple](classTNtuple.html)) |
| tree142_parallelcoordtrans.C | It displays [TParallelCoord](classTParallelCoord.html) from the same dataset twice. The first time without transparency and the second time with transparency |
| tree143_drawsparse.C | Convert a [THnSparse](classTHnSparse.html) to a [TTree](classTTree.html) using efficient iteration through the [THnSparse](classTHnSparse.html) and draw it using [TParallelCoord](classTParallelCoord.html) |


\anchor tree_other
## More tutorials

A collection of other examples which showcases other usecases
| **Tutorial** | **Description** |
|--------------|-----------------|
| tree200_temperature.C | Illustrates how to use the highlight mode with trees, using a temperature dataset from Prague between 1775 and 2004 |
| tree201_histograms.C | Save histograms in tree branches |
| tree202_benchmarks.C | Benchmark comparing row-wise and column-wise storage performance |
| tree500_cernbuild.C | Read data (CERN staff) from an ascii file and create a root file with a tree |
| tree501_cernstaff.C | Playing with a tree containing variables of type character |
| tree502_staff.C | Create a plot of the data in `cernstaff.root` |


\anchor tree_alltutorials
@}
