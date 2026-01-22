import { useState } from 'react';
import axios from 'axios';
import { Header } from './components/Header';
import { HeroSection } from './components/HeroSection';
import { ResultCard } from './components/ResultCard';

type AppState = 'idle' | 'loading' | 'result' | 'error';

interface Verse {
  chapter: number;
  verse: number;
  sanskrit: string;
  transliteration: string;
  synonyms: string;
  translation: string;
  purport: string;
  interpretation: string;
  keywords: string[];
}

export default function App() {
  const [state, setState] = useState<AppState>('idle');
  const [userInput, setUserInput] = useState('');
  // 1. New state for Chapter Filtering
  const [selectedChapter, setSelectedChapter] = useState<number | null>(null);
  const [selectedVerse, setSelectedVerse] = useState<Verse | null>(null);

  const handleSeekGuidance = async () => {
    if (!userInput.trim()) return;

    setState('loading');

    try {
      // Connects to the FastAPI backend
      const response = await axios.post('https://cowsyy-anugamana-backend.hf.space/search', {
        query: userInput,
        limit: 1,
        // 2. Send the selected chapter (if any) to the backend
        chapter: selectedChapter
      });

      const data = response.data.results[0];

      if (data) {
        
        const rawEmotions = data.metadata.emotions || ""; 
        const emotionTags = rawEmotions.split(',').map((s: string) => s.trim()).filter(Boolean);

        // 3. AI ADVICE LOGIC: 
        // Check if the backend provided 'ai_advice' (Gemini RAG). 
        // If yes, use it. If no, fall back to the template.
        const aiAdvice = data.metadata.ai_advice;
        
        const finalInterpretation = aiAdvice 
          ? aiAdvice 
          : (emotionTags.length > 0 
              ? `This verse specifically addresses feelings of ${emotionTags[0]} and offers spiritual guidance on how to navigate them.`
              : 'This verse resonates with your current state of mind. Reflect on its meaning to find clarity.');

        const verseData: Verse = {
          chapter: data.metadata.chapter,
          verse: data.metadata.verse,
          sanskrit: data.metadata.sanskrit,
          transliteration: data.metadata.transliteration || '',
          synonyms: data.metadata.synonyms || '',
          translation: data.metadata.translation || data.text,
          purport: data.metadata.purport || '',
          
          interpretation: finalInterpretation, // Use the smart interpretation
          keywords: emotionTags 
        };

        setSelectedVerse(verseData);
        setState('result');
      } else {
        console.warn("No results found.");
        setState('error');
      }

    } catch (error) {
      console.error("Error connecting to backend:", error);
      setState('error');
    }
  };

  const handleSearchAgain = () => {
    setState('idle');
    setUserInput('');
    setSelectedVerse(null);
    // Note: We deliberately do NOT reset selectedChapter here, 
    // so the user can ask multiple questions within the same chapter context.
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-amber-50 to-orange-50">
      <Header showBackButton={state === 'result'} onBack={handleSearchAgain} />
      
      {state === 'result' && selectedVerse ? (
        <ResultCard
          verse={selectedVerse}
          onSearchAgain={handleSearchAgain}
          userInput={userInput}
        />
      ) : state === 'error' ? (
         <main className="container mx-auto px-4 flex flex-col items-center justify-center min-h-[calc(100vh-80px)]">
            <div className="text-center space-y-4">
              <h2 className="text-2xl font-serif text-red-700">Connection Error</h2>
              <p className="text-stone-600">
                Could not reach the wisdom engine. Is the backend server running?
              </p>
              <button 
                onClick={handleSearchAgain}
                className="px-6 py-2 bg-orange-700 text-white rounded-full hover:bg-orange-800 transition-colors"
              >
                Try Again
              </button>
            </div>
         </main>
      ) : (
        <main className="container mx-auto px-4 flex items-center justify-center min-h-[calc(100vh-80px)]">
          <div className="w-full max-w-3xl">
            {/* 4. Pass the new props to HeroSection */}
            <HeroSection
              state={state === 'loading' ? 'loading' : 'idle'}
              userInput={userInput}
              onInputChange={setUserInput}
              onSeekGuidance={handleSeekGuidance}
              selectedChapter={selectedChapter}
              onChapterChange={setSelectedChapter}
            />
          </div>
        </main>
      )}
    </div>
  );
}