import { Loader2 } from 'lucide-react';
import { motion } from 'motion/react';
import { useState, useEffect } from 'react';

type AppState = 'idle' | 'loading' | 'result';

interface HeroSectionProps {
  state: AppState;
  userInput: string;
  onInputChange: (value: string) => void;
  onSeekGuidance: () => void;
}

const EXAMPLE_PROMPTS = [
  "I'm seeking purpose and direction in my life...",
  "I feel overwhelmed by too many choices and decisions...",
  "I'm struggling to let go of past mistakes...",
  "I want to find inner peace and calm my anxious mind...",
  "I'm unsure which path to take in my career...",
  "I'm seeking courage to face my fears...",
  "I want to understand how to deal with change...",
  "I'm looking for guidance on spiritual growth...",
  "I want to learn how to focus better and avoid distractions...",
  "I'm trying to overcome comparison with others...",
];

export function HeroSection({
  state,
  userInput,
  onInputChange,
  onSeekGuidance,
}: HeroSectionProps) {
  const [placeholderIndex, setPlaceholderIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setPlaceholderIndex((prev) => (prev + 1) % EXAMPLE_PROMPTS.length);
    }, 3000); // Change every 3 seconds

    return () => clearInterval(interval);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      onSeekGuidance();
    }
  };

  return (
    <motion.div
      className="text-center"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-4xl md:text-5xl font-light text-orange-950 mb-4">
          Find clarity through the Bhagavad Gita
        </h2>
        <p className="text-lg text-orange-800 mb-8">
          Describe your current state of mind, confusion, or dilemma
        </p>
      </motion.div>

      <div className="space-y-4">
        <div className="relative">
          <textarea
            value={userInput}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={state === 'loading'}
            placeholder={EXAMPLE_PROMPTS[placeholderIndex]}
            className="w-full h-40 px-6 py-4 rounded-2xl border-2 border-orange-200 focus:border-orange-400 focus:outline-none resize-none text-lg bg-white/80 backdrop-blur-sm disabled:bg-gray-100 disabled:text-gray-500 transition-all placeholder:transition-opacity placeholder:duration-500"
            aria-label="Describe your dilemma"
          />
        </div>

        <button
          onClick={onSeekGuidance}
          disabled={state === 'loading' || !userInput.trim()}
          className="w-full md:w-auto px-12 py-4 bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-700 hover:to-orange-800 text-white rounded-full text-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 mx-auto shadow-lg hover:shadow-xl"
        >
          {state === 'loading' ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Analyzing...</span>
            </>
          ) : (
            <span>Seek Guidance</span>
          )}
        </button>

        {state === 'loading' && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-sm text-orange-700 italic"
          >
            Scanning the wisdom of 700 verses...
          </motion.p>
        )}
      </div>
    </motion.div>
  );
}