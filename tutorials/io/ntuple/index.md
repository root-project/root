\defgroup tutorial_ntuple RNTuple tutorials
\ingroup tutorial_io
\brief Various examples demonstrating ROOT's RNTuple columnar I/O subsystem.

RNTuple is the evolution of TTree, offering better performance and new, more robust interfaces. It additionally comes with a binary format specification, [which can be found here](md_tree_2ntuple_2v7_2doc_2BinaryFormatSpecification.html). The tutorials and examples on this page demonstrate how the RNTuple interface can be used for various use cases.

| **Tutorial** | **Description** |
|--------------|-----------------|
| ntpl001_staff.C | Write and read tabular data with the RNTupleWriter and RNTupleReader |
| ntpl002_vector.C | Write and read vector-based data with the RNTupleWriter and RNTupleReader |
| ntpl004_dimuon.C | Analyze data stored in RNTuple with RDataFrame |
| ntpl005_introspection.C | Write and read data from a user-defined class with the RNTupleWriter and RNTupleReader, and collect runtime I/O information using RNTupleMetrics |
| ntpl007_mtFill.C | Fill multiple entries in parallel  with the RNTupleWriter |
| ntpl008_import.C | Convert data stored in TTree to RNTuple with the RNTupleImporter |
| ntpl009_parallelWriter.C | Write multithreaded with the RNTupleParallelWriter |
| ntpl010_skim.C | Create a derived RNTuple (dropping, renaming and adding fields, applying cuts) with the RNTupleReader and RNTupleWriter |
| ntpl012_processor_chain.C | Read entries from a chain of RNTuples in a single event loop with the RNTupleProcessor |
| ntpl013_staged.C | Apply staged cluster committing to multithreaded writing with the RNTupleParallelWriter |
| ntpl014_framework.C | Use the various (more advanced) RNTuple interfaces to write data in the context of a framework |
| ntpl015_processor_join.C | Join the entries from two RNTuples on a common field value and read it using the RNTupleProcessor |
