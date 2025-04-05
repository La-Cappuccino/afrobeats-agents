# Afrobeats.no Code Standards

This document outlines the coding standards and best practices for the Afrobeats.no codebase.

## Directory Structure

```
afrobeats-agents/
├── agents/                 # Agent implementations
│   ├── coordinator_agent.py
│   ├── dj_booking_agent.py
│   ├── event_discovery_agent.py
│   ├── playlist_agent.py
│   ├── dj_rating_agent.py
│   ├── content_agent.py
│   ├── social_media_agent.py
│   ├── analytics_agent.py
│   └── utils/              # Shared agent utilities
│       ├── state.py
│       ├── prompts.py
│       └── validators.py
├── lib/                    # Shared libraries
│   ├── gemini.py           # Gemini API client
│   ├── spotify.py          # Spotify API client
│   ├── supabase.py         # Supabase client
│   └── n8n.py              # n8n workflow client
├── services/               # Business logic services
│   ├── dj_service.py
│   ├── event_service.py
│   ├── playlist_service.py
│   ├── rating_service.py
│   ├── content_service.py
│   └── social_media_service.py
├── models/                 # Data models (Pydantic)
│   ├── agent_models.py
│   ├── dj_models.py
│   ├── event_models.py
│   ├── playlist_models.py
│   └── user_models.py
├── api/                    # API routes
│   ├── dj.py
│   ├── event.py
│   ├── playlist.py
│   ├── rating.py
│   ├── auth.py
│   └── webhooks/
│       ├── n8n.py
│       └── spotify.py
├── ml/                     # Machine learning models
│   ├── recommendation/
│   │   ├── dj_recommender.py
│   │   ├── event_recommender.py
│   │   └── playlist_recommender.py
│   ├── nlp/
│   │   ├── intent_classifier.py
│   │   ├── sentiment_analyzer.py
│   │   └── entity_extractor.py
│   └── embeddings/
│       ├── song_embeddings.py
│       └── dj_embeddings.py
├── tests/                  # Test suite
│   ├── agents/
│   ├── services/
│   ├── api/
│   └── ml/
├── scripts/                # Utility scripts
│   ├── data_migration.py
│   ├── seed_database.py
│   └── generate_embeddings.py
├── config/                 # Configuration files
│   ├── default.py
│   ├── development.py
│   └── production.py
├── docs/                   # Documentation
├── agent_graph.py          # Agent graph definition
├── run.py                  # Main entry point
└── README.md               # Project overview
```

## Naming Conventions

### General

- Use descriptive names that reflect the purpose of variables, functions, classes, etc.
- Follow the language-specific conventions (PEP 8 for Python, StandardJS for JavaScript)

### Python

- **Files**: Lowercase with underscores (snake_case): `dj_booking_agent.py`
- **Classes**: CapitalizedWords (PascalCase): `DJBookingAgent`
- **Functions/Methods**: Lowercase with underscores (snake_case): `process_booking_request()`
- **Variables**: Lowercase with underscores (snake_case): `booking_details`
- **Constants**: All uppercase with underscores: `MAX_BOOKING_DAYS`
- **Private Methods/Variables**: Prefix with single underscore: `_validate_input()`

### JavaScript/TypeScript

- **Files**: Lowercase with hyphens (kebab-case): `dj-booking.js`
- **Components**: CapitalizedWords (PascalCase): `DJProfileCard.jsx`
- **Functions**: camelCase: `processBookingRequest()`
- **Variables**: camelCase: `bookingDetails`
- **Constants**: All uppercase with underscores: `MAX_BOOKING_DAYS`
- **Private Methods/Properties**: No specific prefix (use TypeScript's private keyword)

## Code Style

### Python

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Use docstrings for all functions, classes, and modules
- Maximum line length of 88 characters (Black formatter standard)
- Use f-strings for string formatting

```python
def get_available_djs(
    date: str, 
    genres: List[str] = None, 
    location: str = "Oslo"
) -> List[Dict[str, Any]]:
    """
    Get a list of DJs available on a specific date.
    
    Args:
        date: Date in YYYY-MM-DD format
        genres: Optional list of genres to filter by
        location: Location to search in (default: Oslo)
        
    Returns:
        List of available DJ objects
    """
    # Implementation
    pass
```

### JavaScript/TypeScript

- Use ESLint with Airbnb config
- Use JSDoc for documentation
- Prefer const over let, avoid var
- Use optional chaining and nullish coalescing
- Use destructuring where appropriate
- Use async/await over promises
- Use TypeScript interfaces for complex objects

```typescript
/**
 * Get a list of DJs available on a specific date
 * @param {string} date - Date in YYYY-MM-DD format
 * @param {string[]} [genres] - Optional list of genres to filter by
 * @param {string} [location="Oslo"] - Location to search in
 * @returns {Promise<DJ[]>} List of available DJ objects
 */
async function getAvailableDJs(
  date: string,
  genres?: string[],
  location: string = "Oslo"
): Promise<DJ[]> {
  // Implementation
}
```

## Agent Implementation Standards

### System Prompts

- Store all system prompts as constants in separate files
- Use templates with clear parameter placeholders
- Include version numbers in prompt names for tracking changes
- Document the purpose and expected output of each prompt

```python
# prompts.py
DJ_BOOKING_SYSTEM_PROMPT_V1 = """
You are the DJ Booking Agent for Afrobeats.no, specializing in helping users 
book Afrobeats and Amapiano DJs in Oslo, Norway.

Your core responsibilities:
1. Process DJ booking requests for events in Oslo
2. Answer questions about the booking process
3. Provide information about available DJs and their ratings
...
"""
```

### Error Handling

- Use structured error handling with appropriate exception types
- Log errors with reference IDs for tracking
- Return sanitized error messages to users
- Include contextual information in error logs

```python
try:
    # Attempt the operation
    result = process_booking(booking_data)
    return result
except ValidationError as e:
    # Handle validation errors
    error_id = f"ERR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.warning(f"Validation error {error_id}: {str(e)}")
    return {"error": f"Invalid booking data. Reference: {error_id}"}
except Exception as e:
    # Handle unexpected errors
    error_id = f"ERR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.error(f"Unexpected error {error_id}: {str(e)}")
    logger.debug(traceback.format_exc())
    return {"error": f"An unexpected error occurred. Reference: {error_id}"}
```

### Agent State Management

- Use typed dictionaries for agent state
- Include validation for state transitions
- Separate state concerns by domain
- Avoid global state

```python
class BookingState(TypedDict):
    user_id: str
    dj_id: Optional[str]
    date: str
    event_type: str
    location: str
    status: Literal["pending", "confirmed", "declined"]

def transition_booking_state(
    current_state: BookingState, 
    action: str, 
    data: Dict[str, Any]
) -> BookingState:
    """
    Transition the booking state based on an action.
    
    Args:
        current_state: The current booking state
        action: The action to perform (confirm, decline, etc.)
        data: Additional data for the transition
        
    Returns:
        The new booking state
    """
    # Implementation
    pass
```

## Security Standards

### Authentication

- Use Supabase Auth for user authentication
- Implement role-based access control (RBAC)
- Use short-lived JWT tokens
- Store tokens securely (httpOnly cookies)
- Implement proper session management

### API Security

- Validate all inputs using Pydantic models
- Sanitize user-generated content
- Use HTTPS for all communications
- Implement rate limiting
- Set appropriate CORS headers

### Sensitive Data

- Never log sensitive information (personal data, credentials)
- Use environment variables for secrets
- Don't store API keys in code
- Encrypt sensitive data at rest
- Implement proper data access controls in Supabase

```python
# Correct way to handle API keys
api_key = os.environ.get("SPOTIFY_API_KEY")
if not api_key:
    raise EnvironmentError("SPOTIFY_API_KEY environment variable is not set")

# Never do this
api_key = "sk_live_1234567890abcdef"  # WRONG!
```

## Testing Standards

### Unit Tests

- Use pytest for Python tests
- Use Jest for JavaScript tests
- Aim for 80%+ test coverage
- Test all public interfaces
- Use mocks for external dependencies

```python
def test_dj_availability_check():
    # Arrange
    dj_id = "test-dj-id"
    date = "2025-05-01"
    mock_supabase = MagicMock()
    mock_supabase.from().select().eq().eq().execute.return_value = {
        "data": [{"id": 1, "start_time": "18:00", "end_time": "22:00"}],
        "error": None
    }
    
    # Act
    result = check_dj_availability(dj_id, date, mock_supabase)
    
    # Assert
    assert result["is_available"] is True
    assert len(result["availability"]) == 1
```

### Integration Tests

- Test API endpoints with real Supabase test instance
- Test agent interactions with controlled inputs
- Use fixtures for setup and teardown
- Test all error cases

### End-to-End Tests

- Test core user flows
- Use Playwright for browser-based testing
- Test multilingual functionality
- Test responsive design

## Performance Standards

### Database Access

- Use indexes for frequent queries
- Minimize N+1 query problems with proper joins
- Use pagination for large result sets
- Optimize expensive queries
- Use caching for frequent queries

### LLM Usage

- Cache LLM responses when appropriate
- Use streaming responses for long-form content
- Optimize prompt engineering for efficiency
- Implement fallback strategies for API failures
- Monitor token usage and costs

```python
def get_cached_or_new_response(prompt, cache_key):
    # Check cache first
    cached_response = redis_client.get(cache_key)
    if cached_response:
        return json.loads(cached_response)
    
    # Generate new response
    response = generate_llm_response(prompt)
    
    # Cache the response with expiration
    redis_client.set(cache_key, json.dumps(response), ex=3600)  # 1 hour
    
    return response
```

## Machine Learning Standards

### Data Pipeline

- Document data sources and transformations
- Use feature stores for reproducibility
- Validate data quality
- Handle outliers and missing data
- Version control datasets

### Model Training

- Document model architecture and hyperparameters
- Track experiments with metrics
- Implement cross-validation
- Test models for bias and fairness
- Version control models

### Model Deployment

- Deploy models as separate services
- Implement monitoring for drift
- Provide fallback mechanisms
- Document API interfaces
- Implement A/B testing infrastructure

## Documentation Standards

### Code Documentation

- All functions should have docstrings
- Complex algorithms should have detailed explanations
- Include examples for non-obvious usages
- Document assumptions and edge cases
- Update documentation when code changes

### API Documentation

- Use OpenAPI/Swagger for API documentation
- Document request/response schemas
- Include authentication requirements
- Provide example requests
- Document error responses

### Architecture Documentation

- Maintain up-to-date architecture diagrams
- Document system dependencies
- Document integration points
- Include deployment architecture
- Keep decision logs for major architectural choices