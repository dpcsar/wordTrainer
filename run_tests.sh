#!/bin/bash

# Run all tests in the tests directory

echo "Running all unit tests..."
python -m unittest discover -s tests -p "test_*.py" -v

UNIT_TEST_RESULT=$?

echo -e "\nRunning non-keywords test..."
python tests/test_non_keywords.py

NON_KEYWORDS_TEST_RESULT=$?

if [ $UNIT_TEST_RESULT -eq 0 ] && [ $NON_KEYWORDS_TEST_RESULT -eq 0 ]; then
  echo -e "\n✅ All tests passed!"
  exit 0
else
  echo -e "\n❌ Some tests failed."
  exit 1
fi
