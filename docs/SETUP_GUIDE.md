# Afrobeats.no Setup Guide

This guide provides step-by-step instructions for setting up the Afrobeats.no agent system.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.10 or higher
- Node.js 18 or higher (for Next.js integration)
- Git
- A Supabase account
- An OpenAI API key or Gemini API key
- A Spotify Developer account
- An n8n instance (self-hosted or cloud)

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/La-Cappuccino/afrobeats-agents.git
cd afrobeats-agents
```

### 2. Create a Virtual Environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n afrobeats-agents python=3.10
conda activate afrobeats-agents
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
# .env
# API Keys
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# n8n Configuration
N8N_WEBHOOK_URL=your_n8n_webhook_url
N8N_WEBHOOK_SECRET=your_n8n_webhook_secret

# Application Configuration
APP_URL=http://localhost:3000
ENVIRONMENT=development
LOG_LEVEL=debug
```

## Database Setup

### 1. Configure Supabase Project

1. Create a new Supabase project
2. Note your project URL and API keys

### 2. Set Up Database Schema

Run the following commands to set up the database schema:

```bash
# Install Supabase CLI
npm install -g supabase

# Login to Supabase
supabase login

# Link your project
supabase link --project-ref your_project_reference

# Run migrations
supabase db push
```

Alternatively, you can manually set up the database schema by running the SQL scripts in the `supabase/migrations` directory through the Supabase SQL editor.

### 3. Enable Extensions

Enable the following extensions in your Supabase project:

- `vector` - For ML vector storage
- `pg_stat_statements` - For query performance monitoring

You can enable these through the Supabase dashboard under Database > Extensions.

## API Integration Setup

### 1. Configure Spotify API

1. Create a Spotify Developer account: https://developer.spotify.com/
2. Create a new application
3. Set the redirect URI to `{APP_URL}/api/auth/spotify/callback`
4. Note your Client ID and Client Secret

### 2. Set Up n8n

1. Set up an n8n instance (self-hosted or cloud)
2. Create the required workflow templates:
   - Social media posting workflow
   - Event notification workflow
   - Content scheduling workflow
3. Set up webhook endpoints in n8n
4. Note your webhook URLs and secrets

### 3. Gemini API Setup (Alternative to OpenAI)

1. Create a Google Cloud account if you don't have one
2. Enable the Vertex AI API
3. Create API credentials and note your API key

## Running the Agent System

### 1. Start the Agent System

```bash
python run.py
```

This will start the agent system, which will:
- Initialize all agents
- Set up the agent graph
- Listen for queries

### 2. Test the Agent System

You can test the agent system with a simple query:

```bash
python test_agent.py --query "Find me an Afrobeats DJ available next Saturday in Oslo"
```

### 3. Running in Development Mode

For development, you can run the system with debugging enabled:

```bash
python run.py --debug
```

This will:
- Enable verbose logging
- Show all agent interactions
- Stream agent responses

## Next.js Integration

### 1. Set Up Next.js App

1. Create a new Next.js app:

```bash
npx create-next-app@latest app-afrotorget
cd app-afrotorget
```

2. Install required dependencies:

```bash
npm install @supabase/supabase-js @supabase/auth-helpers-nextjs axios
```

3. Configure environment variables:

Create a `.env.local` file:

```bash
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
AGENT_API_URL=http://localhost:5000
```

### 2. Create API Routes

Create an API route to proxy requests to the agent system:

```javascript
// app/api/agent/route.js
import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

export async function POST(request) {
  try {
    const { query, userId } = await request.json();
    
    // Get user profile
    let userProfile = null;
    if (userId) {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', userId)
        .single();
        
      if (!error) {
        userProfile = data;
      }
    }
    
    // Call agent system
    const response = await fetch(process.env.AGENT_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        userProfile
      }),
    });
    
    const data = await response.json();
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error calling agent system:', error);
    return NextResponse.json(
      { error: 'Failed to process your request' },
      { status: 500 }
    );
  }
}
```

### 3. Create Agent Client Hook

Create a React hook to interact with the agent system:

```javascript
// hooks/useAgent.js
import { useState } from 'react';

export function useAgent() {
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const sendQuery = async (query, userId = null) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch('/api/agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          userId
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get response from agent');
      }
      
      const data = await response.json();
      setResponse(data);
      return data;
    } catch (error) {
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };
  
  return {
    response,
    isLoading,
    error,
    sendQuery
  };
}
```

## Deploying to Production

### 1. Deploying the Agent System

The agent system can be deployed as a serverless function on Vercel:

1. Create a `vercel.json` file:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "agent_api.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "agent_api.py"
    }
  ]
}
```

2. Create a simplified `agent_api.py` file:

```python
from flask import Flask, request, jsonify
from run import run_agent_graph

app = Flask(__name__)

@app.route('/', methods=['POST'])
def agent_endpoint():
    data = request.json
    query = data.get('query', '')
    user_profile = data.get('userProfile')
    
    result = run_agent_graph(query, user_profile=user_profile)
    
    return jsonify(result)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
```

3. Deploy to Vercel:

```bash
vercel
```

### 2. Deploying Next.js Apps

Deploy Next.js apps to Vercel:

```bash
vercel
```

### 3. Setting Up Production Environment

For production, ensure:

1. All API keys are set up as environment variables in Vercel
2. Rate limiting is configured
3. Monitoring is set up
4. Error tracking is configured

## Monitoring and Maintenance

### 1. Logging

The system uses structured logging:

```python
import logging
import json
from datetime import datetime

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add JSON handler for production
    if level <= logging.INFO:
        json_handler = logging.FileHandler('logs/agent_system.json')
        json_handler.setFormatter(JsonFormatter())
        logging.getLogger().addHandler(json_handler)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'agent'):
            log_record['agent'] = record.agent
            
        if hasattr(record, 'query'):
            log_record['query'] = record.query
            
        if hasattr(record, 'duration'):
            log_record['duration'] = record.duration
            
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)
```

### 2. Monitoring

Set up monitoring for:

1. LLM API usage and costs
2. Agent performance
3. Database performance
4. API response times

### 3. Maintenance Tasks

Regular maintenance tasks include:

1. Updating LLM prompts
2. Retraining ML models
3. Updating database indexes
4. Reviewing logs for errors

## Troubleshooting

### Common Issues

1. **LLM API Rate Limiting**

   If you encounter rate limiting with OpenAI or Gemini:
   
   - Implement backoff and retry logic
   - Consider using a queue system for high-volume scenarios
   - Use local caching for common queries

2. **Database Performance**

   If Supabase queries are slow:
   
   - Check query plans
   - Review indexes
   - Optimize complex queries
   - Consider materialized views for complex reporting

3. **Agent Routing Issues**

   If agents aren't routing properly:
   
   - Check the coordinator agent prompt
   - Review agent logs for reasoning
   - Adjust thresholds for intent classification

4. **Next.js Integration Issues**

   If the Next.js app can't communicate with the agent system:
   
   - Check CORS settings
   - Verify API routes
   - Check environment variables
   - Review network logs for errors

## Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Supabase Documentation](https://supabase.com/docs)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [n8n Documentation](https://docs.n8n.io/)
- [Vercel Documentation](https://vercel.com/docs)