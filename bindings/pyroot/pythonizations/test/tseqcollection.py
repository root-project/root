import unittest

import ROOT


class TSeqCollectionItemAccess(unittest.TestCase):
    """
    Test for the item access methods added to TSeqCollection (and subclasses):
    __getitem__, __setitem__, __delitem__.
    Both the index (l[i]) and slice (l[i:j:k]) syntaxes are tested.
    """

    num_elems = 3

    _global_objects = []

    # Helpers
    def create_tseqcollection(self):
        sc = ROOT.TList()
        for _ in range(self.num_elems):
            o = ROOT.TObject()
            sc.Add(o)
            # To prevent deletion of the objects (TList is by default non-owning)
            self._global_objects.append(o)

        return sc

    # Tests
    def test_getitem(self):
        sc = self.create_tseqcollection()

        # Get items
        it = ROOT.TIter(sc)
        for i in range(self.num_elems):
            self.assertIs(it.Next(), sc[i])

        # Get items, negative indices
        it2 = ROOT.TIter(sc)
        neg_idcs = [ -i-1 for i in reversed(range(self.num_elems)) ]
        for i in neg_idcs:
            self.assertIs(it2.Next(), sc[i])

        # Check invalid index cases
        with self.assertRaises(IndexError):
            sc[self.num_elems]

        with self.assertRaises(IndexError):
            sc[-(self.num_elems + 1)]

        with self.assertRaises(TypeError):
            sc[1.0]

    def test_getitem_slice(self):
        sc = self.create_tseqcollection()

        # All items
        slice1 = sc[:]
        for i in range(slice1.GetEntries()):
            self.assertIs(sc[i], slice1[i])

        # First two items
        slice2 = sc[0:2]
        self.assertIs(sc[0], slice2[0])
        self.assertIs(sc[1], slice2[1])

        # Last two items
        slice3 = sc[-2:]
        self.assertIs(sc[1], slice3[0])
        self.assertIs(sc[2], slice3[1])

        # First and third items
        slice4 = sc[0::2]
        self.assertIs(sc[0], slice4[0])
        self.assertIs(sc[2], slice4[1])

        # All items, reverse order
        slice5 = sc[::-1]
        for i in range(slice5.GetEntries()):
            self.assertIs(sc[i], slice5[self.num_elems - 1 - i])

        # First and third items, reverse order
        slice6 = sc[::-2]
        self.assertIs(sc[0], slice6[1])
        self.assertIs(sc[2], slice6[0])

        # Step cannot be zero
        with self.assertRaises(ValueError):
            sc[::0]

    def test_setitem(self):
        sc = self.create_tseqcollection()
        l1 = []

        # Set items
        for i in range(self.num_elems):
            o = ROOT.TObject()
            sc[i] = o
            l1.append(o)

        # Check previously set items
        it = ROOT.TIter(sc)
        for i in range(self.num_elems):
            self.assertIs(it.Next(), l1[i])

        # Set items, negative indices
        l2 = []
        neg_idcs = [ -i-1 for i in reversed(range(self.num_elems)) ]
        for i in neg_idcs:
            o = ROOT.TObject()
            sc[i] = o
            l2.append(o)

        # Check previously set items
        it2 = ROOT.TIter(sc)
        for i in range(self.num_elems):
            self.assertIs(it2.Next(), l2[i])

        # Check invalid index cases
        with self.assertRaises(IndexError):
            sc[self.num_elems] = ROOT.TObject()

        with self.assertRaises(IndexError):
            sc[-(self.num_elems + 1)] = ROOT.TObject()

        with self.assertRaises(TypeError):
            sc[1.0] = ROOT.TObject()

        # Clear before the added element might be garbage collected,
        # to avoid dangling pointer access.
        sc.Clear()

    def test_setitem_slice(self):
        sc1 = self.create_tseqcollection()
        sc2 = self.create_tseqcollection()

        # Replace all items
        sc1[:] = sc2
        self.assertEqual(sc1.GetEntries(), self.num_elems)
        for i in range(self.num_elems):
            self.assertIs(sc1[i], sc2[i])

        # Append items
        sc1 = self.create_tseqcollection()
        l1 = [elem for elem in sc1]

        sc1[self.num_elems:] = sc2

        self.assertEqual(sc1.GetEntries(), 2 * self.num_elems)
        i = 0
        for elem in l1:  # first half
            self.assertIs(sc1[i], elem)
            i += 1
        for elem in sc2:  # second half
            self.assertIs(sc1[i], elem)
            i += 1

        # Assign second item.
        # This time use a Python list as assigned value
        sc3 = self.create_tseqcollection()
        l2 = [ ROOT.TObject() ]
        l3 = [ elem for elem in sc3 ]

        sc3[1:2] = l2

        self.assertEqual(sc3.GetEntries(), self.num_elems)
        self.assertIs(sc3[0], l3[0])
        self.assertIs(sc3[1], l2[0])
        self.assertIs(sc3[2], l3[2])

        # Assign second and third items to just one item.
        # This tests that the third item is removed
        sc4 = self.create_tseqcollection()
        l4 = [ ROOT.TObject() ]
        l5 = [ elem for elem in sc4 ]

        sc4[1:3] = l4

        self.assertEqual(sc4.GetEntries(), self.num_elems - 1)
        self.assertIs(sc4[0], l5[0])
        self.assertIs(sc4[1], l4[0])

        # Assign with step
        sc5 = self.create_tseqcollection()
        o = sc5[1]
        len6 = 2
        l6 = [ ROOT.TObject() for _ in range(len6) ]

        sc5[::2] = l6

        self.assertEqual(sc5.GetEntries(), self.num_elems)
        self.assertIs(sc5[0], l6[0])
        self.assertIs(sc5[1], o)
        self.assertIs(sc5[2], l6[1])

        # Assign with step (start from end)
        sc5[::-2] = l6

        self.assertEqual(sc5.GetEntries(), self.num_elems)
        self.assertIs(sc5[0], l6[1])
        self.assertIs(sc5[1], o)
        self.assertIs(sc5[2], l6[0])

        # Step cannot be zero
        sc6 = self.create_tseqcollection()
        with self.assertRaises(ValueError):
            sc6[::0] = [ ROOT.TObject() ]

    def test_delitem(self):
        sc = self.create_tseqcollection()

        self.assertEqual(sc.GetEntries(), self.num_elems)

        # Delete all elements
        for _ in range(self.num_elems):
            del sc[0]

        self.assertEqual(sc.GetEntries(), 0)

        sc = ROOT.TList()
        o1 = ROOT.TObject()
        o2 = ROOT.TObject()
        sc.Add(o1)
        sc.Add(o2)
        sc.Add(o1)

        # Delete o2
        del sc[1]

        # Only o1s should be there
        self.assertEqual(sc.GetEntries(), 2)

        it = ROOT.TIter(sc)
        for _ in range(2):
            self.assertIs(it.Next(), o1)

        # Check invalid index cases
        with self.assertRaises(IndexError):
            del sc[2]

        with self.assertRaises(IndexError):
            del sc[-3]

        with self.assertRaises(TypeError):
            del sc[1.0]

        sc.Clear()

    def test_delitem_slice(self):
        # Delete all items
        sc1 = self.create_tseqcollection()
        del sc1[:]
        self.assertEqual(sc1.GetEntries(), 0)

        # Do not delete anything (slice out of range)
        sc2 = self.create_tseqcollection()
        l2 = [ elem for elem in sc2 ]
        del sc2[self.num_elems:]
        self.assertEqual(sc2.GetEntries(), self.num_elems)
        for el1, el2 in zip(sc2, l2):
            self.assertIs(el1, el2)

        # Delete first two items
        sc3 = self.create_tseqcollection()
        o = sc3[2]
        del sc3[0:2]
        self.assertEqual(sc3.GetEntries(), 1)
        self.assertIs(sc3[0], o)

        # Delete first and third items
        sc4 = self.create_tseqcollection()
        o = sc4[1]
        del sc4[::2]
        self.assertEqual(sc4.GetEntries(), 1)
        self.assertIs(sc4[0], o)

        # Delete first and third items (start from end)
        sc5 = self.create_tseqcollection()
        o = sc5[1]
        del sc5[::-2]
        self.assertEqual(sc5.GetEntries(), 1)
        self.assertIs(sc5[0], o)

        # Step cannot be zero
        sc6 = self.create_tseqcollection()
        with self.assertRaises(ValueError):
            sc6[::0]



class TSeqCollectionListMethods(unittest.TestCase):
    """
    Test for the Python-list-like methods added to TSeqCollection
    (and subclasses): insert, pop, reverse, sort, index
    """

    num_elems = 3

    _global_objects = []

    # Helpers
    def create_tseqcollection(self):
        sc = ROOT.TList()
        for i in reversed(range(self.num_elems)):
            o = ROOT.TObjString(str(i))
            sc.Add(o)
            # To prevent deletion of the objects (TList is by default non-owning)
            self._global_objects.append(o)

        return sc

    # Tests
    def test_insert(self):
        sc = self.create_tseqcollection()

        # Insert with positive index
        o1 = ROOT.TObject()
        sc.insert(1, o1)
        self.assertEqual(sc.GetEntries(), self.num_elems + 1)
        self.assertIs(sc.At(1), o1)

        # Insert with negative index (starts from end)
        o2 = ROOT.TObject()
        sc.insert(-1, o2)
        self.assertEqual(sc.GetEntries(), self.num_elems + 2)
        self.assertIs(sc.At(self.num_elems), o2)

        # Insert with index beyond lower boundary.
        # Inserts at the beginning
        o3 = ROOT.TObject()
        sc.insert(-(self.num_elems + 3), o3)
        self.assertEqual(sc.GetEntries(), self.num_elems + 3)
        self.assertIs(sc.At(0), o3)

        # Insert with index beyond upper boundary.
        # Inserts at the end
        o4 = ROOT.TObject()
        sc.insert(self.num_elems + 4, o4)
        self.assertEqual(sc.GetEntries(), self.num_elems + 4)
        self.assertIs(sc.At(self.num_elems + 3), o4)

        # Clear before the added element might be garbage collected,
        # to avoid dangling pointer access.
        sc.Clear()

    def test_pop(self):
        sc = self.create_tseqcollection()
        l1 = [elem for elem in sc]

        # No arguments, pop last item
        self.assertEqual(sc.pop(), l1[-1])
        self.assertEqual(sc.GetEntries(), self.num_elems - 1)

        # Pop first item, positive index
        self.assertEqual(sc.pop(0), l1[0])
        self.assertEqual(sc.GetEntries(), self.num_elems - 2)

        # Pop last item, negative index
        self.assertEqual(sc.pop(-1), l1[1])
        self.assertEqual(sc.GetEntries(), self.num_elems - 3)

        # Pop from empty collection
        with self.assertRaises(IndexError):
            sc.pop()

        # Index out of range, positive
        sc2 = self.create_tseqcollection()
        with self.assertRaises(IndexError):
            sc2.pop(self.num_elems)

        # Index out of range, negative
        with self.assertRaises(IndexError):
            sc2.pop(-(self.num_elems + 1))

        # Pop with non-integer argument
        with self.assertRaises(TypeError):
            sc2.pop(1.0)

        # Pop a repeated element.
        # Keep Python reference so the added element lives beyond the sc2.pop() call:
        new_elem = ROOT.TObjString("2")
        sc2.append(new_elem)
        elem = sc2.pop()
        self.assertEqual(sc2.At(0), elem)

    def test_reverse(self):
        sc = self.create_tseqcollection()
        l1 = [elem for elem in sc]

        sc.reverse()

        self.assertEqual(sc.GetEntries(), self.num_elems)
        for i,elem in zip(range(self.num_elems), sc):
            self.assertEqual(elem, l1[-(i + 1)])

        # Empty collection
        sc2 = ROOT.TList()
        sc2.reverse()
        self.assertEqual(sc2.GetEntries(), 0)

    def test_sort(self):
        sc = self.create_tseqcollection()
        l1 = [elem for elem in sc]

        # Regular sort, rely on TList::Sort
        sc.sort()
        # We need to set `key` until the pythonization to
        # make TObjString comparable is there
        l1.sort(key=lambda s: s.GetName())

        self.assertEqual(sc.GetEntries(), self.num_elems)
        self.assertEqual(l1[0], sc[0])
        for el1, el2 in zip(sc, l1):
            self.assertEqual(el1, el2)

        # Python sort, key and reverse arguments.
        # Sort by hash in reverse order
        sc2 = self.create_tseqcollection()
        l2 = [ elem for elem in sc2 ]

        def fsort(elem):
            return elem.Hash()

        rev = True
        sc2.sort(key = fsort, reverse = rev)
        l2.sort(key = fsort, reverse = rev)

        self.assertEqual(sc2.GetEntries(), self.num_elems)
        for el1, el2 in zip(sc2, l2):
            self.assertEqual(el1, el2)

        # Empty collection
        sc4 = ROOT.TList()
        sc4.sort()
        self.assertEqual(sc4.GetEntries(), 0)

    def test_index(self):
        sc = self.create_tseqcollection()

        # Check all elements of collection
        for i, elem in zip(range(self.num_elems), sc):
            self.assertEqual(sc.index(elem), i)

        # Check element not in collection
        o1 = ROOT.TObjString(str(self.num_elems))
        with self.assertRaises(ValueError):
            sc.index(o1)


if __name__ == '__main__':
    unittest.main()
