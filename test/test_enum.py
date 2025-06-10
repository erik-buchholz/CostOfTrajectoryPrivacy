#!/usr/bin/env python3
import unittest
from privtrajgen.config.enums import CustomEnum


class TestEnum(CustomEnum):
    NAME_A = 'value a'
    NAME_B = 'value b'
    NAME_C = 'value c'

class TestCustomEnum(unittest.TestCase):
    def test_enum(self):
        value = TestEnum.NAME_A
        # Verify that string comparison works for both the name and the value, but enum comparison works too
        self.assertEqual(value, 'value a')
        self.assertEqual(value, 'NAME_A')
        self.assertEqual(value, TestEnum.NAME_A)
        self.assertNotEqual(value, 'value b')
        # Verify that str return the name and not Class.NAME
        self.assertEqual(str(value), 'NAME_A')
        # Test printing
        self.assertEqual(f'{value}', 'NAME_A')

if __name__ == '__main__':
    unittest.main()
