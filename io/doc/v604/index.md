## I/O Libraries

### I/O New functionalities

- Support for forward_list and I/O of unordered stl containers.
- Support for std::complex.

### I/O Behavior change.

- The I/O now properly skip the content of base class onfile that have been removed from the in-memory class layout.

- The scheduling the I/O customization rules within a StreamerInfo is now as soon as possible, i.e. after all sources have been read.  One significant consequence is that now when an object is stored in a split branch
the rule is associtated with the branch of the last of the rule's sources rather
than the last of the object's data member.

- Properly support TStreamerInfo written by ROOT v4.00.
