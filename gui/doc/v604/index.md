
## GUI Libraries

### TGTextViewostream

- A new `TGTextViewostream` class has been added. It is a text viewer widget and is a specialization of `TGTextView` and `std::ostream`. It uses a `TGTextViewStreamBuf`, which inherits from `std::streambuf`, allowing to stream text directly to the text view in a `cout` - like fashion. A new tutorial showing how to use the `TGTextViewostream` widget has also been added.
