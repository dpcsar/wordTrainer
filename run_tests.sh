#!/bin/bash

# Run all tests in the tests directory

echo "Running all tests..."
python -m unittest discover -s tests -p "test_*.py" -v

if [ $? -eq 0 ]; then
  echo -e "\n✅ All tests passed!"
else
  echo -e "\n❌ Some tests failed."
  exit 1
fi
