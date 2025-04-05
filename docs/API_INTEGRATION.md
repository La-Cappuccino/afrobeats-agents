# Afrobeats.no API Integration

This document outlines the API integrations used by the Afrobeats.no platform, focusing on:
- Spotify API for playlist functionality
- Gemini API for LLM capabilities
- n8n for workflow automation
- Supabase API for database operations

## Spotify API Integration

### Authentication Flow

The platform uses Spotify's OAuth 2.0 implementation for authentication:

```javascript
// auth/spotify.js
import { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabase';

const SPOTIFY_CLIENT_ID = process.env.NEXT_PUBLIC_SPOTIFY_CLIENT_ID;
const REDIRECT_URI = `${process.env.NEXT_PUBLIC_APP_URL}/api/auth/spotify/callback`;
const SCOPES = [
  'user-read-private',
  'user-read-email',
  'playlist-read-private',
  'playlist-read-collaborative',
  'playlist-modify-public',
  'playlist-modify-private',
  'user-library-read'
];

export function useSpotifyAuth() {
  const [session, setSession] = useState(null);
  const [spotifyToken, setSpotifyToken] = useState(null);
  
  useEffect(() => {
    // Get session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      
      // Get Spotify token from session
      if (session?.provider_token) {
        setSpotifyToken(session.provider_token);
      }
    });
    
    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
        setSpotifyToken(session?.provider_token);
      }
    );
    
    return () => subscription.unsubscribe();
  }, []);
  
  const login = () => {
    const authUrl = `https://accounts.spotify.com/authorize?client_id=${SPOTIFY_CLIENT_ID}&response_type=code&redirect_uri=${encodeURIComponent(REDIRECT_URI)}&scope=${encodeURIComponent(SCOPES.join(' '))}`;
    window.location.href = authUrl;
  };
  
  return { session, spotifyToken, login };
}
```

### Playlist Operations

The platform uses Spotify's API for playlist management:

```javascript
// lib/spotify.js
import SpotifyWebApi from 'spotify-web-api-node';

export class SpotifyService {
  constructor(accessToken) {
    this.spotifyApi = new SpotifyWebApi({
      clientId: process.env.SPOTIFY_CLIENT_ID,
      clientSecret: process.env.SPOTIFY_CLIENT_SECRET
    });
    
    if (accessToken) {
      this.spotifyApi.setAccessToken(accessToken);
    }
  }
  
  async createPlaylist(name, description, isCollaborative = false, isPublic = true) {
    try {
      const data = await this.spotifyApi.createPlaylist(name, {
        description,
        public: isPublic,
        collaborative: isCollaborative
      });
      
      return data.body;
    } catch (error) {
      console.error('Error creating playlist:', error);
      throw error;
    }
  }
  
  async addTracksToPlaylist(playlistId, trackUris) {
    try {
      const data = await this.spotifyApi.addTracksToPlaylist(playlistId, trackUris);
      return data.body;
    } catch (error) {
      console.error('Error adding tracks to playlist:', error);
      throw error;
    }
  }
  
  async getPlaylist(playlistId) {
    try {
      const data = await this.spotifyApi.getPlaylist(playlistId);
      return data.body;
    } catch (error) {
      console.error('Error getting playlist:', error);
      throw error;
    }
  }
  
  async searchTracks(query, limit = 20) {
    try {
      const data = await this.spotifyApi.searchTracks(query, { limit });
      return data.body.tracks.items;
    } catch (error) {
      console.error('Error searching tracks:', error);
      throw error;
    }
  }
  
  async getMyPlaylists() {
    try {
      const data = await this.spotifyApi.getUserPlaylists();
      return data.body.items;
    } catch (error) {
      console.error('Error getting user playlists:', error);
      throw error;
    }
  }
  
  async getTrackFeatures(trackId) {
    try {
      const data = await this.spotifyApi.getAudioFeaturesForTrack(trackId);
      return data.body;
    } catch (error) {
      console.error('Error getting track features:', error);
      throw error;
    }
  }
  
  // This method converts Spotify data to our database schema
  formatPlaylistForDatabase(spotifyPlaylist, userId) {
    return {
      name: spotifyPlaylist.name,
      description: spotifyPlaylist.description || '',
      creator_id: userId,
      image_url: spotifyPlaylist.images[0]?.url || null,
      is_collaborative: spotifyPlaylist.collaborative,
      spotify_id: spotifyPlaylist.id,
      is_private: !spotifyPlaylist.public,
      genres: [], // We'll need to derive this from the tracks
      vote_count: 0
    };
  }
  
  formatTrackForDatabase(spotifyTrack) {
    return {
      title: spotifyTrack.name,
      artist: spotifyTrack.artists.map(a => a.name).join(', '),
      album: spotifyTrack.album.name,
      release_date: spotifyTrack.album.release_date,
      spotify_id: spotifyTrack.id,
      youtube_id: null, // We'd need a separate API for this
      genre: [], // Spotify doesn't provide genre at track level
      popularity: spotifyTrack.popularity
    };
  }
}
```

### Playlist Synchronization

The platform synchronizes playlists between Spotify and our database:

```javascript
// agents/services/playlist_sync.js
import { supabase } from '@/lib/supabase';
import { SpotifyService } from '@/lib/spotify';

export async function syncPlaylistWithSpotify(playlistId, accessToken) {
  const spotifyService = new SpotifyService(accessToken);
  
  // Get playlist from database
  const { data: playlist, error } = await supabase
    .from('playlists')
    .select('*')
    .eq('id', playlistId)
    .single();
  
  if (error) {
    throw new Error(`Error fetching playlist: ${error.message}`);
  }
  
  // If playlist already has Spotify ID, update it
  if (playlist.spotify_id) {
    // Get tracks from database
    const { data: playlistTracks, error: tracksError } = await supabase
      .from('playlist_tracks')
      .select(`
        *,
        songs:song_id (*)
      `)
      .eq('playlist_id', playlistId)
      .order('position');
    
    if (tracksError) {
      throw new Error(`Error fetching playlist tracks: ${tracksError.message}`);
    }
    
    // Update Spotify playlist
    await spotifyService.replaceTracksInPlaylist(
      playlist.spotify_id,
      playlistTracks.map(pt => `spotify:track:${pt.songs.spotify_id}`)
    );
    
    return playlist.spotify_id;
  } else {
    // Create new Spotify playlist
    const spotifyPlaylist = await spotifyService.createPlaylist(
      playlist.name,
      playlist.description,
      playlist.is_collaborative,
      !playlist.is_private
    );
    
    // Get tracks from database
    const { data: playlistTracks, error: tracksError } = await supabase
      .from('playlist_tracks')
      .select(`
        *,
        songs:song_id (*)
      `)
      .eq('playlist_id', playlistId)
      .order('position');
    
    if (tracksError) {
      throw new Error(`Error fetching playlist tracks: ${tracksError.message}`);
    }
    
    // Add tracks to Spotify playlist
    if (playlistTracks.length > 0) {
      await spotifyService.addTracksToPlaylist(
        spotifyPlaylist.id,
        playlistTracks.map(pt => `spotify:track:${pt.songs.spotify_id}`)
      );
    }
    
    // Update database with Spotify ID
    await supabase
      .from('playlists')
      .update({ spotify_id: spotifyPlaylist.id })
      .eq('id', playlistId);
    
    return spotifyPlaylist.id;
  }
}
```

## Gemini API Integration

The platform uses Google's Gemini 2.5 for AI capabilities:

```python
# lib/gemini.py
import os
import google.generativeai as genai
from typing import Dict, List, Any

# Configure the Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class GeminiService:
    def __init__(self, model_name="gemini-2.5-pro"):
        self.model = genai.GenerativeModel(model_name)
    
    async def generate_text(self, prompt: str, system_prompt: str = None, temperature: float = 0.7):
        """
        Generate text using Gemini
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt to guide the model
            temperature: Creativity temperature (0.0-1.0)
            
        Returns:
            Generated text response
        """
        try:
            generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
            }
            
            chat = self.model.start_chat(history=[])
            
            if system_prompt:
                # Add system prompt to guide the model
                chat.send_message(system_prompt, role="system")
            
            response = chat.send_message(prompt, generation_config=generation_config)
            return response.text
            
        except Exception as e:
            print(f"Error generating text with Gemini: {str(e)}")
            # Fallback to a simpler prompt if the initial one fails
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except:
                return "I encountered an error processing your request. Please try again with simpler instructions."
    
    async def extract_structured_data(self, text: str, schema: Dict[str, Any]):
        """
        Extract structured data from text based on a schema
        
        Args:
            text: Text to extract data from
            schema: Dictionary defining the schema
            
        Returns:
            Structured data as a dictionary
        """
        try:
            prompt = f"""
            Extract the following information from this text:
            {text}
            
            The response should be a JSON object with these fields:
            {schema}
            
            Only return valid JSON, no other text.
            """
            
            response = await self.generate_text(prompt, temperature=0.1)
            
            # Remove any markdown formatting and extract the JSON
            response = response.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON - in actual code you'd want to use a JSON parser
            import json
            return json.loads(response)
            
        except Exception as e:
            print(f"Error extracting structured data: {str(e)}")
            return {"error": "Failed to extract structured data"}
    
    async def generate_social_media_content(self, details: Dict[str, Any]):
        """
        Generate social media content based on provided details
        
        Args:
            details: Dictionary with details for the social media post
            
        Returns:
            Formatted social media content
        """
        platform = details.get("platform", "instagram")
        event = details.get("event")
        dj = details.get("dj")
        
        prompt = f"""
        Create engaging social media content for {platform} about Afrobeats/Amapiano music in Oslo, Norway.
        
        Details:
        {details}
        
        Generate both English and Norwegian versions.
        Include appropriate hashtags, emojis, and formatting for {platform}.
        Make it authentic to the Afrobeats community.
        
        Format the response as a JSON object with fields:
        - content_en: English version
        - content_no: Norwegian version
        - hashtags: List of hashtags
        - emojis: List of suggested emojis
        """
        
        system_prompt = "You are a social media specialist for Afrobeats.no, creating engaging content for the Afrobeats and Amapiano community in Oslo, Norway."
        
        response = await self.generate_text(prompt, system_prompt, temperature=0.8)
        
        # Extract JSON response
        try:
            import json
            response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(response)
        except:
            # Fallback with an error
            return {
                "content_en": "Check out our latest Afrobeats events in Oslo!",
                "content_no": "Sjekk ut våre nyeste Afrobeats-arrangementer i Oslo!",
                "hashtags": ["#Afrobeats", "#Oslo", "#Amapiano"],
                "emojis": ["🎵", "🔥", "💃"]
            }
```

## n8n Workflow Integration

The platform uses n8n for workflow automation:

### Workflow Triggers

```javascript
// api/webhooks/n8n.js
import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

export async function POST(request) {
  try {
    const { webhookKey } = request.query;
    const secretKey = process.env.N8N_WEBHOOK_SECRET;
    
    // Verify webhook secret
    if (webhookKey !== secretKey) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }
    
    const body = await request.json();
    const { eventType, data } = body;
    
    // Handle different event types
    switch (eventType) {
      case 'playlist_created':
        await handlePlaylistCreated(data);
        break;
      case 'dj_booked':
        await handleDjBooked(data);
        break;
      case 'event_created':
        await handleEventCreated(data);
        break;
      case 'social_post_scheduled':
        await handleSocialPostScheduled(data);
        break;
      default:
        return NextResponse.json(
          { error: 'Unknown event type' },
          { status: 400 }
        );
    }
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Webhook error:', error);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}

async function handlePlaylistCreated(data) {
  // Logic to handle a new playlist being created
  // This could trigger social media posts, etc.
}

async function handleDjBooked(data) {
  // Logic to handle a DJ being booked
  // This could send notifications, update calendars, etc.
}

async function handleEventCreated(data) {
  // Logic to handle a new event being created
  // This could generate social media content, etc.
}

async function handleSocialPostScheduled(data) {
  // Logic to handle a social post being scheduled
  // This could update analytics, etc.
}
```

### Example n8n Workflow JSON

The following is an example of an n8n workflow configuration for social media automation:

```json
{
  "name": "Social Media Post Generator",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [
            {
              "field": "days",
              "minutesInterval": 1440
            }
          ]
        }
      },
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "typeVersion": 1,
      "position": [
        250,
        300
      ]
    },
    {
      "parameters": {
        "url": "https://app.afrotorget.no/api/agent/social-media/generate",
        "options": {
          "response": {
            "response": {
              "fullResponse": true
            }
          }
        }
      },
      "name": "Generate Social Post",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 1,
      "position": [
        450,
        300
      ]
    },
    {
      "parameters": {
        "functionCode": "return {\n  instagram: {\n    content: $input.item.json.data.content_en,\n    contentNo: $input.item.json.data.content_no,\n    hashtags: $input.item.json.data.hashtags.join(' '),\n    platform: 'instagram'\n  },\n  twitter: {\n    content: $input.item.json.data.content_en.substring(0, 240),\n    contentNo: $input.item.json.data.content_no.substring(0, 240),\n    hashtags: $input.item.json.data.hashtags.slice(0, 3).join(' '),\n    platform: 'twitter'\n  }\n};"
      },
      "name": "Format for Platforms",
      "type": "n8n-nodes-base.function",
      "typeVersion": 1,
      "position": [
        650,
        300
      ]
    },
    {
      "parameters": {
        "url": "https://api.instagram.com/v1/media/upload",
        "authentication": "oAuth2",
        "method": "POST",
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "caption",
              "value": "={{ $node[\"Format for Platforms\"].json[\"instagram\"].content + \"\\n\\n\" + $node[\"Format for Platforms\"].json[\"instagram\"].hashtags }}"
            }
          ]
        }
      },
      "name": "Post to Instagram",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 1,
      "position": [
        850,
        200
      ],
      "credentials": {
        "oAuth2Api": {
          "id": "1",
          "name": "Instagram OAuth2 account"
        }
      }
    },
    {
      "parameters": {
        "url": "https://api.twitter.com/2/tweets",
        "authentication": "oAuth2",
        "method": "POST",
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "text",
              "value": "={{ $node[\"Format for Platforms\"].json[\"twitter\"].content + \"\\n\\n\" + $node[\"Format for Platforms\"].json[\"twitter\"].hashtags }}"
            }
          ]
        }
      },
      "name": "Post to Twitter",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 1,
      "position": [
        850,
        400
      ],
      "credentials": {
        "oAuth2Api": {
          "id": "2",
          "name": "Twitter OAuth2 account"
        }
      }
    },
    {
      "parameters": {
        "url": "https://app.afrotorget.no/api/webhooks/n8n?webhookKey={{ $env.N8N_WEBHOOK_SECRET }}",
        "method": "POST",
        "sendBody": true,
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "eventType",
              "value": "social_post_published"
            },
            {
              "name": "data",
              "value": "={{ {instagram: $node[\"Post to Instagram\"].json, twitter: $node[\"Post to Twitter\"].json} }}"
            }
          ]
        }
      },
      "name": "Callback to App",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 1,
      "position": [
        1050,
        300
      ]
    }
  ],
  "connections": {
    "Schedule Trigger": {
      "main": [
        [
          {
            "node": "Generate Social Post",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Generate Social Post": {
      "main": [
        [
          {
            "node": "Format for Platforms",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Format for Platforms": {
      "main": [
        [
          {
            "node": "Post to Instagram",
            "type": "main",
            "index": 0
          },
          {
            "node": "Post to Twitter",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Post to Instagram": {
      "main": [
        [
          {
            "node": "Callback to App",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Post to Twitter": {
      "main": [
        [
          {
            "node": "Callback to App",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

## Supabase API Integration

The platform uses Supabase for database operations:

### Supabase Client Setup

```javascript
// lib/supabase.js
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Server-side client with service role for admin operations
export const createServiceClient = () => {
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  return createClient(supabaseUrl, supabaseServiceKey);
};
```

### Database Operations

```javascript
// services/dj-service.js
import { supabase } from '@/lib/supabase';

export async function getDJs(filters = {}) {
  try {
    let query = supabase
      .from('dj_profiles')
      .select(`
        *,
        profiles (id, display_name, avatar_url),
        avg_rating:dj_reviews(rating).avg()
      `);
    
    // Apply filters
    if (filters.genres && filters.genres.length > 0) {
      query = query.containedBy('genres', filters.genres);
    }
    
    if (filters.minRating) {
      query = query.gte('avg_rating', filters.minRating);
    }
    
    if (filters.available === true) {
      query = query.eq('available_for_booking', true);
    }
    
    const { data, error } = await query;
    
    if (error) {
      throw error;
    }
    
    return data;
  } catch (error) {
    console.error('Error fetching DJs:', error);
    throw error;
  }
}

export async function getDJById(id) {
  try {
    const { data, error } = await supabase
      .from('dj_profiles')
      .select(`
        *,
        profiles (id, display_name, avatar_url, bio, location),
        dj_reviews (
          id,
          rating,
          review_text,
          created_at,
          reviewer:profiles (id, display_name, avatar_url)
        ),
        avg_rating:dj_reviews(rating).avg(),
        review_count:dj_reviews(id).count()
      `)
      .eq('id', id)
      .single();
    
    if (error) {
      throw error;
    }
    
    return data;
  } catch (error) {
    console.error(`Error fetching DJ with ID ${id}:`, error);
    throw error;
  }
}

export async function checkDJAvailability(djId, date) {
  try {
    // Check if the DJ has availability on this date
    const { data: availability, error: availabilityError } = await supabase
      .from('dj_availability')
      .select('*')
      .eq('dj_id', djId)
      .eq('date', date);
    
    if (availabilityError) {
      throw availabilityError;
    }
    
    // Check if the DJ already has bookings on this date
    const { data: bookings, error: bookingsError } = await supabase
      .from('bookings')
      .select('*')
      .eq('dj_id', djId)
      .eq('date', date)
      .in('status', ['pending', 'confirmed']);
    
    if (bookingsError) {
      throw bookingsError;
    }
    
    // Determine if the DJ is available
    const hasAvailability = availability.length > 0;
    const hasConflictingBookings = bookings.length > 0;
    
    return {
      isAvailable: hasAvailability && !hasConflictingBookings,
      availability,
      bookings
    };
  } catch (error) {
    console.error(`Error checking availability for DJ ${djId} on ${date}:`, error);
    throw error;
  }
}
```

### Supabase Auth Integration

```javascript
// auth/supabase.js
import { supabase } from '@/lib/supabase';
import { useState, useEffect } from 'react';

export function useSupabaseAuth() {
  const [session, setSession] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user || null);
      setLoading(false);
    });
    
    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
        setUser(session?.user || null);
        setLoading(false);
      }
    );
    
    return () => subscription.unsubscribe();
  }, []);
  
  const signIn = async ({ email, password }) => {
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password
      });
      
      if (error) {
        throw error;
      }
      
      return data;
    } catch (error) {
      console.error('Error signing in:', error);
      throw error;
    }
  };
  
  const signUp = async ({ email, password, name }) => {
    try {
      // Create the user
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            display_name: name
          }
        }
      });
      
      if (error) {
        throw error;
      }
      
      // Create profile
      if (data.user) {
        await supabase.from('profiles').insert({
          id: data.user.id,
          display_name: name,
          email
        });
      }
      
      return data;
    } catch (error) {
      console.error('Error signing up:', error);
      throw error;
    }
  };
  
  const signOut = async () => {
    try {
      const { error } = await supabase.auth.signOut();
      
      if (error) {
        throw error;
      }
    } catch (error) {
      console.error('Error signing out:', error);
      throw error;
    }
  };
  
  return {
    session,
    user,
    loading,
    signIn,
    signUp,
    signOut
  };
}
```

## Security Best Practices

The platform follows these security best practices for API integration:

1. **Environment Variables**
   - All API keys stored in environment variables
   - Different keys for development and production

2. **CORS Configuration**
   - Restrictive CORS policy for API endpoints
   - Allowlisting only necessary domains

3. **Rate Limiting**
   - Rate limiting on all API endpoints
   - Graduated rate limits based on user authentication

4. **Input Validation**
   - Schema validation for all API inputs
   - Sanitization of user inputs

5. **Token Handling**
   - Refresh token rotation
   - Short-lived access tokens
   - Secure storage of tokens in httpOnly cookies

6. **Error Handling**
   - Sanitized error messages
   - Detailed logging for debugging
   - Generic messages for users

7. **Webhook Security**
   - Secret key verification for webhooks
   - Signature validation for n8n webhooks