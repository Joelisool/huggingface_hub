-- Run this to update your conversations.db structure

ALTER TABLE conversations ADD COLUMN local_context TEXT;
ALTER TABLE emotional_growth ADD COLUMN local_processing_data TEXT;
ALTER TABLE personality_growth ADD COLUMN local_model_feedback TEXT;

-- Add indices for better local search performance
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
CREATE INDEX IF NOT EXISTS idx_emotional_growth_timestamp ON emotional_growth(timestamp);
