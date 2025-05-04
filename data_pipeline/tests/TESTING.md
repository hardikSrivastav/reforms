# Testing the Data Pipeline

This document provides guidance on testing the data pipeline module, including best practices, patterns, and troubleshooting.

## Testing Philosophy

The data pipeline test suite follows these key principles:

1. **Isolation**: Unit tests should test one component in isolation, mocking all dependencies.
2. **Integration**: Integration tests should verify that components work together correctly.
3. **Reproducibility**: Tests should be deterministic and not rely on external services.
4. **Coverage**: Tests should cover both happy paths and error scenarios.

## Test Structure

The test suite is organized into:

```
tests/
├── __init__.py       # Package marker
├── conftest.py       # Shared fixtures and configuration
├── unit/             # Unit tests (testing isolated components)
│   ├── __init__.py
│   ├── test_embedding_service.py
│   ├── test_qdrant_service.py
│   └── ...
├── integration/      # Integration tests (testing component interactions)
│   ├── __init__.py
│   ├── test_process_survey.py
│   └── ...
└── TESTING.md        # This documentation file
```

## Fixtures

The test suite includes several fixtures (in `conftest.py`) that provide mock implementations of external dependencies:

- `mock_qdrant_client`: Mocks the Qdrant client
- `mock_qdrant_service`: Mocks the QdrantService
- `mock_openai_client`: Mocks the OpenAI client
- `mock_embedding_service`: Mocks the EmbeddingService
- `mock_mongodb_client`: Mocks the MongoDB client
- `sample_survey_data`: Provides sample survey data
- `sample_survey_responses`: Provides sample survey responses
- `sample_metrics`: Provides sample metrics

Use these fixtures in your tests to avoid depending on external services.

## Writing Tests

### Unit Tests

When writing unit tests:

1. Use the appropriate fixtures to mock dependencies
2. Test one specific function or method
3. Test both successful and error paths
4. Verify that the function interacts correctly with its dependencies

Example:

```python
@pytest.mark.asyncio
async def test_my_function(mock_dependency):
    # Arrange
    expected_result = "expected output"
    mock_dependency.some_method.return_value = expected_result
    
    # Act
    result = await my_function()
    
    # Assert
    assert result == expected_result
    mock_dependency.some_method.assert_called_once()
```

### Integration Tests

When writing integration tests:

1. Mock external services, but test real interactions between components
2. Focus on data flow through multiple components
3. Verify end-to-end behavior
4. Use patches to inject mock dependencies

Example:

```python
@pytest.mark.asyncio
async def test_end_to_end_flow(mock_service_a, mock_service_b):
    # Arrange
    test_input = "test input"
    expected_output = "expected output"
    
    # Configure mocks
    mock_service_a.process.return_value = "intermediate result"
    mock_service_b.transform.return_value = expected_output
    
    # Act
    with patch('module.get_service_a', return_value=mock_service_a),
         patch('module.get_service_b', return_value=mock_service_b):
        result = await end_to_end_function(test_input)
    
    # Assert
    assert result == expected_output
    mock_service_a.process.assert_called_once_with(test_input)
    mock_service_b.transform.assert_called_once()
```

## Mocking Asyncio

Many components in the data pipeline use asyncio. When mocking async functions:

1. Use the `AsyncMock` class from `unittest.mock`
2. For custom behavior, define an async function and assign it to the mock

Example:

```python
# Simple mock that returns a value
mock_function = AsyncMock(return_value="result")

# Mock with custom behavior
async def custom_mock(*args, **kwargs):
    # Custom logic here
    return "result"

mock_object.method = custom_mock
```

## Running Tests

To run the full test suite:

```bash
./run_tests.sh
```

To run only unit tests:

```bash
python -m pytest tests/unit
```

To run only integration tests:

```bash
python -m pytest tests/integration
```

To run a specific test file:

```bash
python -m pytest tests/unit/test_specific_file.py
```

To run a specific test:

```bash
python -m pytest tests/unit/test_file.py::test_specific_function
```

## Troubleshooting

Common issues and solutions:

1. **Test hangs**: This is often due to unresolved asyncio futures. Make sure all async functions are properly awaited.

2. **Mock not called**: Verify that you're testing the correct function and that the function is actually calling the dependency.

3. **Test interference**: Tests should not depend on each other. If tests interfere, check for shared mutable state.

4. **Fixture errors**: If fixtures don't work as expected, check the fixture scopes and dependencies.

## Adding New Test Cases

When implementing new features:

1. Add unit tests for all new functions and methods
2. Update integration tests to cover new component interactions
3. Ensure tests cover both success and error cases
4. Document any new fixtures in `conftest.py`

## Code Coverage

The test suite aims for high code coverage. To check coverage:

```bash
python -m pytest --cov=data_pipeline tests/
```

This will show which lines and branches are covered by tests. 