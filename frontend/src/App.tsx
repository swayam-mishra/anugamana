import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import Header from './components/Header';
import HeroSection from './components/HeroSection';
import ResultCard from './components/ResultCard';

// Define the API call function outside the component
const fetchVerses = async (searchQuery: string) => {
  // Use your actual backend URL (from Vercel/Render/HF) or localhost for dev
  const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
  
  const response = await fetch(`${apiUrl}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: searchQuery, limit: 5 }),
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  return response.json();
};

function App() {
  const [query, setQuery] = useState('');

  // React Query useMutation handles all the heavy lifting
  const searchMutation = useMutation({
    mutationFn: fetchVerses,
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    // Trigger the mutation
    searchMutation.mutate(query);
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <HeroSection />
        
        {/* Search Form */}
        <form onSubmit={handleSearch} className="max-w-2xl mx-auto mb-12 flex gap-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question..."
            className="flex-1 px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-orange-500 focus:border-transparent outline-none"
          />
          <button 
            type="submit" 
            disabled={searchMutation.isPending}
            className="bg-orange-600 hover:bg-orange-700 text-white px-8 py-3 rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {searchMutation.isPending ? 'Searching...' : 'Search'}
          </button>
        </form>

        {/* Status Messages */}
        {searchMutation.isError && (
          <div className="text-red-500 text-center mb-8">
            An error occurred: {searchMutation.error.message}
          </div>
        )}

        {/* Results */}
        <div className="max-w-4xl mx-auto space-y-6">
          {searchMutation.data?.results?.map((result: any, index: number) => (
            <ResultCard key={index} data={result} />
          ))}
        </div>
      </main>
    </div>
  );
}

export default App;