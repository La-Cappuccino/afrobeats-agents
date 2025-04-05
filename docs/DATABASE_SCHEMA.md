# Afrobeats.no Database Schema

This document outlines the database schema used by the Afrobeats.no platform, implemented in Supabase PostgreSQL.

## Overview

The database is structured to support the core features of Afrobeats.no:
- User accounts and profiles
- DJ profiles and bookings
- Events
- Playlists and voting
- Song information and rankings
- DJ ratings and reviews
- Community forum
- Social media content

## Schema Details

### Authentication and Users

```sql
-- Users table (managed by Supabase Auth)
-- referenced as auth.users

-- Profiles table (extends auth.users)
CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id),
  display_name TEXT NOT NULL,
  avatar_url TEXT,
  bio TEXT,
  location TEXT DEFAULT 'Oslo',
  website TEXT,
  spotify_username TEXT,
  preferred_genres TEXT[] DEFAULT ARRAY['Afrobeats', 'Amapiano'],
  preferred_language TEXT DEFAULT 'en',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Profiles are viewable by everyone" 
  ON profiles FOR SELECT USING (true);

CREATE POLICY "Users can update their own profile" 
  ON profiles FOR UPDATE USING (auth.uid() = id);
```

### DJ Booking System

```sql
-- DJ Profiles
CREATE TABLE dj_profiles (
  id UUID PRIMARY KEY REFERENCES profiles(id),
  genres TEXT[] NOT NULL,
  years_experience INTEGER NOT NULL,
  hourly_rate_min DECIMAL,
  hourly_rate_max DECIMAL,
  equipment TEXT[],
  specialties TEXT[],
  spotify_playlist_url TEXT,
  soundcloud_url TEXT,
  youtube_url TEXT,
  instagram_handle TEXT,
  available_for_booking BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- DJ Availability
CREATE TABLE dj_availability (
  id SERIAL PRIMARY KEY,
  dj_id UUID REFERENCES dj_profiles(id) NOT NULL,
  date DATE NOT NULL,
  start_time TIME NOT NULL,
  end_time TIME NOT NULL,
  location TEXT DEFAULT 'Oslo',
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  UNIQUE (dj_id, date, start_time, end_time)
);

-- Bookings
CREATE TABLE bookings (
  id SERIAL PRIMARY KEY,
  user_id UUID REFERENCES profiles(id) NOT NULL,
  dj_id UUID REFERENCES dj_profiles(id) NOT NULL,
  event_name TEXT NOT NULL,
  event_type TEXT NOT NULL,
  date DATE NOT NULL,
  start_time TIME NOT NULL,
  end_time TIME NOT NULL,
  location TEXT NOT NULL,
  address TEXT,
  expected_guests INTEGER,
  budget DECIMAL,
  equipment_notes TEXT,
  music_preferences TEXT,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'declined', 'completed', 'cancelled')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```

### Events

```sql
-- Events
CREATE TABLE events (
  id SERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  date DATE NOT NULL,
  start_time TIME NOT NULL,
  end_time TIME,
  location TEXT NOT NULL,
  address TEXT,
  venue_name TEXT,
  organizer_id UUID REFERENCES profiles(id),
  image_url TEXT,
  ticket_url TEXT,
  price DECIMAL,
  is_free BOOLEAN DEFAULT false,
  genres TEXT[],
  featured_djs UUID[],
  status TEXT NOT NULL DEFAULT 'upcoming' CHECK (status IN ('upcoming', 'ongoing', 'completed', 'cancelled')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Event RSVPs
CREATE TABLE event_rsvps (
  id SERIAL PRIMARY KEY,
  event_id INTEGER REFERENCES events(id) NOT NULL,
  user_id UUID REFERENCES profiles(id) NOT NULL,
  status TEXT NOT NULL DEFAULT 'going' CHECK (status IN ('going', 'interested', 'not_going')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  UNIQUE (event_id, user_id)
);
```

### Playlists and Music

```sql
-- Songs
CREATE TABLE songs (
  id SERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  artist TEXT NOT NULL,
  album TEXT,
  release_date DATE,
  spotify_id TEXT UNIQUE,
  youtube_id TEXT,
  genre TEXT[] NOT NULL,
  popularity INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Playlists
CREATE TABLE playlists (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  creator_id UUID REFERENCES profiles(id) NOT NULL,
  image_url TEXT,
  is_collaborative BOOLEAN DEFAULT false,
  spotify_id TEXT,
  is_private BOOLEAN DEFAULT false,
  genres TEXT[],
  vote_count INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Playlist Tracks (Many-to-Many)
CREATE TABLE playlist_tracks (
  id SERIAL PRIMARY KEY,
  playlist_id INTEGER REFERENCES playlists(id) NOT NULL,
  song_id INTEGER REFERENCES songs(id) NOT NULL,
  position INTEGER NOT NULL,
  added_by UUID REFERENCES profiles(id) NOT NULL,
  added_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  UNIQUE (playlist_id, song_id)
);

-- Playlist Votes
CREATE TABLE playlist_votes (
  id SERIAL PRIMARY KEY,
  playlist_id INTEGER REFERENCES playlists(id) NOT NULL,
  user_id UUID REFERENCES profiles(id) NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  UNIQUE (playlist_id, user_id)
);

-- Song Rankings
CREATE TABLE song_rankings (
  id SERIAL PRIMARY KEY,
  song_id INTEGER REFERENCES songs(id) NOT NULL,
  rank INTEGER NOT NULL,
  previous_rank INTEGER,
  week_number INTEGER NOT NULL,
  year INTEGER NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  UNIQUE (song_id, week_number, year)
);
```

### DJ Ratings and Reviews

```sql
-- DJ Reviews
CREATE TABLE dj_reviews (
  id SERIAL PRIMARY KEY,
  dj_id UUID REFERENCES dj_profiles(id) NOT NULL,
  reviewer_id UUID REFERENCES profiles(id) NOT NULL,
  booking_id INTEGER REFERENCES bookings(id),
  rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
  review_text TEXT,
  is_verified BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  UNIQUE (dj_id, reviewer_id, booking_id)
);
```

### Community Forum

```sql
-- Forum Categories
CREATE TABLE forum_categories (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Forum Posts
CREATE TABLE forum_posts (
  id SERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  author_id UUID REFERENCES profiles(id) NOT NULL,
  category_id INTEGER REFERENCES forum_categories(id) NOT NULL,
  is_pinned BOOLEAN DEFAULT false,
  is_locked BOOLEAN DEFAULT false,
  view_count INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Forum Comments
CREATE TABLE forum_comments (
  id SERIAL PRIMARY KEY,
  post_id INTEGER REFERENCES forum_posts(id) NOT NULL,
  author_id UUID REFERENCES profiles(id) NOT NULL,
  content TEXT NOT NULL,
  parent_id INTEGER REFERENCES forum_comments(id),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```

### Social Media

```sql
-- Social Media Posts
CREATE TABLE social_posts (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  content_no TEXT,  -- Norwegian version
  platform TEXT NOT NULL CHECK (platform IN ('instagram', 'twitter', 'facebook', 'tiktok')),
  image_url TEXT,
  creator_id UUID REFERENCES profiles(id),
  is_scheduled BOOLEAN DEFAULT false,
  scheduled_time TIMESTAMP WITH TIME ZONE,
  status TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'published', 'scheduled', 'failed')),
  related_event_id INTEGER REFERENCES events(id),
  related_playlist_id INTEGER REFERENCES playlists(id),
  related_dj_id UUID REFERENCES dj_profiles(id),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Social Post Analytics
CREATE TABLE social_post_analytics (
  id SERIAL PRIMARY KEY,
  post_id INTEGER REFERENCES social_posts(id) NOT NULL,
  likes INTEGER DEFAULT 0,
  comments INTEGER DEFAULT 0,
  shares INTEGER DEFAULT 0,
  impressions INTEGER DEFAULT 0,
  platform TEXT NOT NULL,
  measured_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```

### Machine Learning Data

```sql
-- User Interactions (for recommendation system)
CREATE TABLE user_interactions (
  id SERIAL PRIMARY KEY,
  user_id UUID REFERENCES profiles(id) NOT NULL,
  interaction_type TEXT NOT NULL CHECK (interaction_type IN ('view', 'like', 'save', 'book', 'attend')),
  dj_id UUID REFERENCES dj_profiles(id),
  event_id INTEGER REFERENCES events(id),
  playlist_id INTEGER REFERENCES playlists(id),
  song_id INTEGER REFERENCES songs(id),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Song Embeddings (for similarity search)
CREATE TABLE song_embeddings (
  id SERIAL PRIMARY KEY,
  song_id INTEGER REFERENCES songs(id) UNIQUE NOT NULL,
  embedding VECTOR(1536),  -- Using pgvector extension
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- DJ Embeddings (for similarity search)
CREATE TABLE dj_embeddings (
  id SERIAL PRIMARY KEY,
  dj_id UUID REFERENCES dj_profiles(id) UNIQUE NOT NULL,
  embedding VECTOR(1536),  -- Using pgvector extension
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```

## Indexes and Optimizations

```sql
-- Indexes for performance optimization
-- Booking queries
CREATE INDEX idx_bookings_dj_id ON bookings(dj_id);
CREATE INDEX idx_bookings_status ON bookings(status);
CREATE INDEX idx_bookings_date ON bookings(date);

-- Event queries
CREATE INDEX idx_events_date ON events(date);
CREATE INDEX idx_events_genres ON events USING GIN(genres);
CREATE INDEX idx_events_status ON events(status);

-- Playlist optimization
CREATE INDEX idx_playlists_creator ON playlists(creator_id);
CREATE INDEX idx_playlists_genres ON playlists USING GIN(genres);
CREATE INDEX idx_playlists_votes ON playlists(vote_count);

-- DJ searches
CREATE INDEX idx_dj_profiles_genres ON dj_profiles USING GIN(genres);
CREATE INDEX idx_dj_availability_date ON dj_availability(date);

-- Forum optimizations
CREATE INDEX idx_forum_posts_category ON forum_posts(category_id);
CREATE INDEX idx_forum_comments_post ON forum_comments(post_id);

-- ML optimization
CREATE INDEX idx_user_interactions_user ON user_interactions(user_id);
CREATE INDEX idx_user_interactions_type_dj ON user_interactions(interaction_type, dj_id) 
  WHERE interaction_type = 'book' AND dj_id IS NOT NULL;
```

## Vector Search Setup (for ML Features)

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create functions for similarity search
CREATE OR REPLACE FUNCTION get_similar_songs(song_id INTEGER, limit_count INTEGER DEFAULT 5)
RETURNS TABLE (
  id INTEGER,
  title TEXT,
  artist TEXT,
  similarity FLOAT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    s.id,
    s.title,
    s.artist,
    1 - (e.embedding <=> (SELECT embedding FROM song_embeddings WHERE song_id = get_similar_songs.song_id))::FLOAT AS similarity
  FROM songs s
  JOIN song_embeddings e ON s.id = e.song_id
  WHERE s.id != get_similar_songs.song_id
  ORDER BY similarity DESC
  LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_similar_djs(dj_id UUID, limit_count INTEGER DEFAULT 5)
RETURNS TABLE (
  id UUID,
  display_name TEXT,
  genres TEXT[],
  similarity FLOAT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    d.id,
    p.display_name,
    d.genres,
    1 - (e.embedding <=> (SELECT embedding FROM dj_embeddings WHERE dj_id = get_similar_djs.dj_id))::FLOAT AS similarity
  FROM dj_profiles d
  JOIN profiles p ON d.id = p.id
  JOIN dj_embeddings e ON d.id = e.dj_id
  WHERE d.id != get_similar_djs.dj_id
  ORDER BY similarity DESC
  LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;
```

## Row Level Security (RLS) Policies

The database implements row-level security to ensure data is accessed properly:

```sql
-- Example RLS policies
-- DJ availability is editable only by the DJ or admins
CREATE POLICY "DJs can update their own availability"
  ON dj_availability FOR UPDATE USING (auth.uid() = dj_id);

-- Bookings viewable by the booker or the booked DJ
CREATE POLICY "Bookings are viewable by involved parties"
  ON bookings FOR SELECT USING (
    auth.uid() = user_id OR 
    auth.uid() = dj_id OR
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
  );

-- Collaborative playlists can be edited by creator and collaborators
CREATE POLICY "Collaborative playlists can be edited by multiple users"
  ON playlist_tracks FOR INSERT USING (
    EXISTS (
      SELECT 1 FROM playlists 
      WHERE id = playlist_id 
      AND (
        creator_id = auth.uid() OR 
        is_collaborative = true
      )
    )
  );
```

## Database Functions and Triggers

The schema includes functions and triggers for automated operations:

```sql
-- Update vote count when a playlist receives a vote
CREATE OR REPLACE FUNCTION update_playlist_vote_count()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE playlists 
    SET vote_count = vote_count + 1
    WHERE id = NEW.playlist_id;
  ELSIF TG_OP = 'DELETE' THEN
    UPDATE playlists 
    SET vote_count = vote_count - 1
    WHERE id = OLD.playlist_id;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_playlist_vote_change
AFTER INSERT OR DELETE ON playlist_votes
FOR EACH ROW EXECUTE FUNCTION update_playlist_vote_count();

-- Update average DJ rating when a new review is added
CREATE OR REPLACE FUNCTION update_dj_average_rating()
RETURNS TRIGGER AS $$
BEGIN
  -- This would be implemented in the dj_profiles table to calculate average rating
  -- In a real implementation this would be more complex
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_dj_review_change
AFTER INSERT OR UPDATE OR DELETE ON dj_reviews
FOR EACH ROW EXECUTE FUNCTION update_dj_average_rating();
```

## Database Migrations

Migrations are managed through Supabase migrations, with each change tracked in version control. For local development, we use the Supabase CLI to handle migrations.