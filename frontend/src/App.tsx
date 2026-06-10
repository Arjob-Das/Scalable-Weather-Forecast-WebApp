import { useState, useEffect } from 'react';
import { WeatherDashboard } from './components/WeatherDashboard';
import { QueryHistory } from './components/QueryHistory';
import type { WeatherQuery } from './components/QueryHistory';

const BACKEND_URL = import.meta.env.VITE_API_URL || (() => {
  const port = window.location.port;
  let backendPort = '8080';
  if (port === '30000') backendPort = '30080';
  else if (port === '30001') backendPort = '30081';
  return `${window.location.protocol}//${window.location.hostname}:${backendPort}`;
})();

function App() {
  const [city, setCity] = useState('');
  const [weatherData, setWeatherData] = useState<any>(null);
  const [history, setHistory] = useState<WeatherQuery[]>([]);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chartToken, setChartToken] = useState<number>(Date.now());
  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);

  const fetchHistory = async () => {
    setHistoryLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/weather/history`);
      if (response.ok) {
        const data = await response.json();
        setHistory(data);
      }
    } catch (err) {
      console.error('Failed to fetch query history:', err);
    } finally {
      setHistoryLoading(false);
    }
  };

  // Autocomplete Suggestions Hook
  useEffect(() => {
    if (!showDropdown || city.trim().length < 2) {
      setSuggestions([]);
      return;
    }
    const delayDebounce = setTimeout(async () => {
      try {
        const res = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=5&language=en&format=json`);
        if (res.ok) {
          const data = await res.json();
          setSuggestions(data.results || []);
        }
      } catch (err) {
        console.error('Failed to fetch suggestions:', err);
      }
    }, 300);
    return () => clearTimeout(delayDebounce);
  }, [city, showDropdown]);

  const handleSearch = async (searchCity: string) => {
    if (!searchCity.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/api/weather/forecast?city=${encodeURIComponent(searchCity)}`);
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Failed to fetch weather data');
      }
      const data = await response.json();
      setWeatherData(data);
      setChartToken(Date.now());
      fetchHistory(); // Refresh history
    } catch (err: any) {
      setError(err.message || 'An error occurred while fetching data.');
      setWeatherData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setShowDropdown(false);
    setSuggestions([]);
    handleSearch(city);
  };

  // Initial load
  useEffect(() => {
    fetchHistory();
    // Load a default city to make the dashboard look stunning right away
    handleSearch('London');
  }, []);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header glass-panel" style={{ padding: '1rem 1.5rem', border: 'none' }}>
        <h1 className="app-logo">
          AERO-ML <span>Weather Intel</span>
        </h1>
        <span className="app-subtitle">Global Pre-trained LSTM Weather Model</span>
      </header>

      {/* Main Panel */}
      <main className="main-grid">
        {/* Left Column: Search & Dashboard */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Search Box */}
          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <form onSubmit={handleFormSubmit} className="search-box">
              <div style={{ position: 'relative', flex: 1 }}>
                <svg
                  className="search-icon-inside"
                  xmlns="http://www.w3.org/2000/svg"
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="11" cy="11" r="8"></circle>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                <input
                  type="text"
                  className="search-input"
                  placeholder="Search for a city (e.g. New York, Tokyo, Berlin)..."
                  value={city}
                  onChange={(e) => {
                    setCity(e.target.value);
                    setShowDropdown(true);
                  }}
                  onFocus={() => setShowDropdown(true)}
                  onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
                  disabled={loading}
                  autoComplete="off"
                />
                
                {suggestions.length > 0 && showDropdown && (
                  <ul className="suggestions-dropdown glass-panel" style={{
                    position: 'absolute',
                    top: '100%',
                    left: 0,
                    right: 0,
                    zIndex: 1000,
                    marginTop: '0.5rem',
                    padding: '0.5rem 0',
                    listStyle: 'none',
                    maxHeight: '220px',
                    overflowY: 'auto',
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    backdropFilter: 'blur(16px)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.4)'
                  }}>
                    {suggestions.map((sug: any) => (
                      <li 
                        key={sug.id}
                        onClick={() => {
                          const name = sug.name + (sug.admin1 ? `, ${sug.admin1}` : '') + (sug.country ? `, ${sug.country}` : '');
                          setCity(name);
                          setShowDropdown(false);
                          setSuggestions([]);
                          handleSearch(name);
                        }}
                        style={{
                          padding: '0.75rem 1.25rem',
                          cursor: 'pointer',
                          color: 'var(--text-primary)',
                          fontSize: '0.9rem',
                          textAlign: 'left',
                          transition: 'background-color 0.2s',
                          borderBottom: '1px solid rgba(255, 255, 255, 0.05)'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.08)'}
                        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                      >
                        <span style={{ fontWeight: 600 }}>{sug.name}</span>
                        {sug.admin1 && <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>, {sug.admin1}</span>}
                        {sug.country && <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}> ({sug.country})</span>}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              <button type="submit" className="search-btn" disabled={loading}>
                {loading ? (
                  <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }}></div>
                ) : (
                  <>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                    Forecast
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Loading, Error or Dashboard */}
          {loading && !weatherData ? (
            <div className="glass-panel loader-container">
              <div className="spinner"></div>
              <div>
                <h3 style={{ marginBottom: '0.25rem', color: 'var(--text-primary)' }}>Running LSTM Inference...</h3>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                  Retrieving last 30 days of daily weather data and predicting forecast using pre-trained PyTorch model.
                </p>
              </div>
            </div>
          ) : error ? (
            <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', borderLeft: '4px solid var(--error)' }}>
              <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--error)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginBottom: '1rem' }}><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              <h3 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>Failed to Fetch Forecast</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>{error}</p>
              <button 
                className="search-btn" 
                style={{ margin: '1rem auto 0', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--card-border)' }}
                onClick={() => setError(null)}
              >
                Clear Error
              </button>
            </div>
          ) : weatherData ? (
            <WeatherDashboard data={weatherData} backendUrl={BACKEND_URL} chartToken={chartToken} />
          ) : (
            <div className="glass-panel empty-state">
              <h3 className="empty-title">No City Selected</h3>
              <p className="empty-desc">Enter a location above to fetch live weather details, train an ML model, and forecast temperatures.</p>
            </div>
          )}
        </div>

        {/* Right Column: Search History */}
        <QueryHistory history={history} onSelectCity={handleSearch} loading={historyLoading} />
      </main>
    </div>
  );
}

export default App;
