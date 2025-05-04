#!/bin/bash

# Set the environment to test
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running data pipeline tests...${NC}"

# Run basic unit tests
echo -e "\n${YELLOW}Running basic unit tests...${NC}"
python -m pytest tests/unit/test_embedding_service.py tests/unit/test_qdrant_service.py -v

# Store the basic unit test result
BASIC_UNIT_RESULT=$?

# Run metadata and analysis unit tests
echo -e "\n${YELLOW}Running metadata and analysis unit tests...${NC}"
python -m pytest tests/unit/test_base_analysis.py tests/unit/test_metadata_store.py -v

# Store the analysis unit test result
ANALYSIS_UNIT_RESULT=$?

# Run visualization tests
echo -e "\n${YELLOW}Running visualization tests...${NC}"
python -m pytest tests/unit/test_visualization_components.py tests/unit/test_base_analysis_visualizations.py -v

# Store the visualization test result
VISUALIZATION_RESULT=$?

# Run metric intelligence tests (Phase 2)
echo -e "\n${YELLOW}Running metric intelligence tests...${NC}"
python -m pytest tests/unit/test_metric_analysis.py tests/unit/test_cross_metric_analysis.py tests/unit/test_trend_analysis.py -v

# Store the metric intelligence test result
METRIC_INTELLIGENCE_RESULT=$?

# Run integration tests
echo -e "\n${YELLOW}Running integration tests...${NC}"
python -m pytest tests/integration -v

# Store the integration test result
INTEGRATION_RESULT=$?

# Print summary
echo -e "\n${YELLOW}Test Summary:${NC}"
if [ $BASIC_UNIT_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Basic unit tests passed${NC}"
else
    echo -e "${RED}✗ Basic unit tests failed${NC}"
fi

if [ $ANALYSIS_UNIT_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Metadata and analysis unit tests passed${NC}"
else
    echo -e "${RED}✗ Metadata and analysis unit tests failed${NC}"
fi

if [ $VISUALIZATION_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Visualization tests passed${NC}"
else
    echo -e "${RED}✗ Visualization tests failed${NC}"
fi

if [ $METRIC_INTELLIGENCE_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Metric intelligence tests passed${NC}"
else
    echo -e "${RED}✗ Metric intelligence tests failed${NC}"
fi

if [ $INTEGRATION_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Integration tests passed${NC}"
else
    echo -e "${RED}✗ Integration tests failed${NC}"
fi

# Exit with error if any test suite failed
if [ $BASIC_UNIT_RESULT -ne 0 ] || [ $ANALYSIS_UNIT_RESULT -ne 0 ] || [ $VISUALIZATION_RESULT -ne 0 ] || [ $METRIC_INTELLIGENCE_RESULT -ne 0 ] || [ $INTEGRATION_RESULT -ne 0 ]; then
    echo -e "\n${RED}Some tests failed.${NC}"
    exit 1
else
    echo -e "\n${GREEN}All tests passed successfully!${NC}"
    exit 0
fi 