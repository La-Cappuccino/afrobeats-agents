# Afrobeats.no Agent Architecture

This document outlines the architecture of the Afrobeats.no agent system, focusing on the multi-agent design pattern using LangGraph, Pydantic, and ML components.

## System Overview

Afrobeats.no is a multi-agent AI system that powers various features across the platform's domains:
- **afrobeats.no**: Main content site
- **app.afrotorget.no**: User portal and authenticated features
- **amapiano.afrotorget.no**: Genre-specific experiences

The agent system provides intelligent capabilities for:
- DJ booking and recommendation
- Event discovery and promotion
- Playlist curation and voting
- DJ ratings and reviews
- Content generation
- Social media automation
- Analytics and insights

## Agent Graph Architecture

The system implements a hub-and-spoke architecture using LangGraph:

```
                     ┌─────────────────┐
                     │                 │
                     │   Coordinator   │
                     │      Agent      │
                     │                 │
                     └─────────────────┘
                             │
           ┌────────────────┼────────────────┐
           │                │                │
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│                 │ │                 │ │                 │
│   DJ Booking    │ │     Event       │ │    Playlist     │
│      Agent      │ │   Discovery     │ │      Agent      │
│                 │ │      Agent      │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
           │                │                │
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│                 │ │                 │ │                 │
│    DJ Rating    │ │     Content     │ │  Social Media   │
│      Agent      │ │      Agent      │ │      Agent      │
│                 │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                             │
                     ┌─────────────────┐
                     │                 │
                     │    Analytics    │
                     │      Agent      │
                     │                 │
                     └─────────────────┘
```

### Key Components

1. **Coordinator Agent**: Central hub that:
   - Analyzes user queries
   - Routes to specialized agents
   - Integrates responses
   - Maintains conversation context

2. **Specialized Agents**:
   - **DJ Booking Agent**: Handles DJ recommendations and booking processes
   - **Event Discovery Agent**: Manages event search and promotion
   - **Playlist Agent**: Handles music curation and voting
   - **DJ Rating Agent**: Processes reviews and ratings
   - **Content Agent**: Generates news and educational content
   - **Social Media Agent**: Creates platform-specific social content
   - **Analytics Agent**: Provides ML-powered insights

3. **State Management**: Uses TypedDict for structured state passing between agents

## Data Flow

1. User query enters the system
2. Coordinator agent analyzes the query intent
3. Query is routed to appropriate specialized agent(s)
4. Specialized agent processes the query
5. Results return to coordinator
6. Coordinator integrates results
7. Final response delivered to user

## Integration Points

### Supabase Integration
- Authentication
- Database operations
- Real-time subscriptions
- Storage

### LLM Integration
- Primary: Gemini 2.5 for agent intelligence
- Backup/Alternative: OpenAI GPT-4o

### Spotify API
- Playlist creation and management
- Track information
- Collaborative playlists

### n8n Workflows
- Social media automation
- Event notifications
- Content scheduling

## Machine Learning Components

The system incorporates several ML models:

1. **Recommendation Engines**:
   - DJ recommendation based on event type and preferences
   - Event recommendation based on user interests
   - Playlist recommendation based on listening history

2. **Natural Language Processing**:
   - Intent classification for routing
   - Entity extraction for structured data
   - Sentiment analysis for reviews

3. **Time Series Analysis**:
   - Trending track detection
   - Event popularity prediction
   - DJ booking pattern recognition

## Security Architecture

1. **Authentication**:
   - Supabase Auth with JWT
   - Role-based access control

2. **Data Protection**:
   - Encrypted API keys via environment variables
   - Input validation and sanitization
   - JSON parsing with error handling

3. **Error Management**:
   - Structured logging
   - Reference IDs for errors
   - Sanitized user-facing error messages

## Multilingual Support

The system supports both Norwegian and English languages through:
- Language detection
- Parallel content generation
- Language-specific templates

## Agent Communication Protocol

Agents communicate through a structured state object with the following components:

```python
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]  # Conversation history
    user_query: str                 # Original user query
    current_agent: str              # Currently active agent
    dj_booking_results: Dict[str, Any]
    event_discovery_results: Dict[str, Any]
    playlist_results: Dict[str, Any]
    dj_rating_results: Dict[str, Any]
    content_results: Dict[str, Any]
    social_media_results: Dict[str, Any]
    analytics_results: Dict[str, Any]
    final_response: str             # Response to user
```

## Deployment Architecture

The agent system is deployed as a serverless application with the following components:

1. **Core Agent Logic**: Deployed on Vercel serverless functions
2. **Database**: Supabase PostgreSQL
3. **Authentication**: Supabase Auth
4. **Storage**: Supabase Storage
5. **Workflow Automation**: n8n.io
6. **Frontend**: Next.js applications deployed on Vercel

## Future Extensions

The architecture is designed to be extensible in the following ways:

1. **Additional Agents**: New specialized agents can be added to the graph
2. **Enhanced ML Models**: More sophisticated recommendation and analysis capabilities
3. **Voice Interface**: Supporting voice queries through a dedicated interface agent
4. **Mobile SDK**: Embedding agent capabilities in mobile applications
5. **Extended Language Support**: Adding more languages beyond Norwegian and English