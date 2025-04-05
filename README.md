# Afrobeats.no Agentic Platform

A multi-agent AI system for Afrobeats.no, focused on Afrobeats and Amapiano DJ booking, event marketing, playlist curation/voting, and DJ ratings in Oslo, Norway.

## Core Features

- **DJ Booking System**: Find and book DJs specializing in Afrobeats and Amapiano
- **Event Discovery**: Discover and promote Afrobeats/Amapiano events in Oslo
- **Playlist Curation**: Create, share, and vote on music playlists
- **Song Rankings**: Track song popularity and trending tracks in the community
- **Artist Tagging**: Connect songs and playlists with artist information
- **DJ Rating System**: Rate and review DJs, find top-rated professionals
- **Content Hub**: News and updates about the Afrobeats/Amapiano scene in Oslo
- **Social Media Integration**: Automatically create and share content across platforms
- **Community Forum**: Discussions and user-generated content that can be highlighted on social media

## Project Architecture

This project uses LangGraph to create a directed graph of specialized agents, each handling specific aspects of the platform:

- **Coordinator Agent**: Analyzes user queries and routes to specialized agents
- **DJ Booking Agent**: Handles DJ booking requests and availability
- **Event Discovery Agent**: Manages event discovery and promotion
- **Playlist Agent**: Handles playlist curation, sharing, and voting
- **DJ Rating Agent**: Processes DJ ratings and reviews
- **Content Agent**: Provides news and educational content
- **Social Media Agent**: Creates and schedules content for social platforms
- **Analytics Agent**: Tracks user behavior and content performance with ML

## Documentation

Comprehensive documentation is available in the `/docs` directory:

- [Architecture Overview](docs/ARCHITECTURE.md) - Detailed system architecture
- [Database Schema](docs/DATABASE_SCHEMA.md) - Supabase database structure
- [API Integration](docs/API_INTEGRATION.md) - External API integration details
- [ML Integration](docs/ML_INTEGRATION.md) - Machine learning components
- [Setup Guide](docs/SETUP_GUIDE.md) - Installation and configuration instructions
- [Code Standards](docs/CODE_STANDARDS.md) - Coding standards and best practices

## Technical Stack

- **LangGraph**: For agent orchestration and state management
- **Pydantic**: For type-safe data structures
- **Gemini API**: For primary language model capabilities (with OpenAI fallback)
- **Supabase**: For database and authentication
- **Next.js**: For the web application
- **n8n**: For workflow automation
- **Machine Learning**: For recommendation systems, trend analysis, and personalization
- **Social Media APIs**: For automated content publishing

## Machine Learning Components

The platform incorporates several ML models:
- **DJ Recommendation Engine**: Hybrid recommender system for personalized DJ suggestions
- **Song Recommendation Engine**: Collaborative filtering for personalized music recommendations
- **Trend Analysis**: Time-series analysis of song popularity and event engagement
- **User Segmentation**: Clustering users based on music preferences and behavior
- **Content Performance Prediction**: Predicting which content will perform well on social media
- **Natural Language Processing**: For intent classification, entity extraction, and sentiment analysis

## Platform Strategy

The system is designed to support three key domains:

- **afrobeats.no**: Main content site
- **app.afrotorget.no**: Next.js application for authenticated features
- **amapiano.afrotorget.no**: Genre-specific subdomain

All three domains are supported by the same agent infrastructure and multilingual capabilities (Norwegian and English).

## Social Media Integration

The platform uses the Social Media Agent to:
- Automatically create posts highlighting new playlists, events, and DJs
- Extract interesting discussions from the community forum
- Promote upcoming events with customized content for each platform
- Track engagement metrics to continuously improve content strategy
- Schedule posts at optimal times based on audience activity

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. Run the agent system:
   ```
   python run.py
   ```

4. For development and testing:
   ```
   python run.py --debug
   ```

5. For integration with Next.js, see the [Setup Guide](docs/SETUP_GUIDE.md).

## Database Setup

Follow these steps to set up the Supabase database:

1. Create a Supabase project
2. Run the migration scripts in `supabase/migrations`
3. Set up proper Row Level Security policies
4. Configure authentication providers

For detailed instructions, see the [Database Schema](docs/DATABASE_SCHEMA.md) documentation.

## Multilingual Support

The platform supports both Norwegian and English languages to serve the diverse Afrobeats/Amapiano community in Norway. Each agent is designed to process queries in either language and respond appropriately.

## Contributing

We welcome contributions to the Afrobeats.no agent system! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.