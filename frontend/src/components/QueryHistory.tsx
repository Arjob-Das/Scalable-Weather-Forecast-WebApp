import React from 'react';

export interface WeatherQuery {
  id: number;
  cityName: string;
  latitude: number;
  longitude: number;
  queriedAt: string;
  tempPredictedMean: number | null;
  description: string;
}

interface QueryHistoryProps {
  history: WeatherQuery[];
  onSelectCity: (cityName: string) => void;
  loading: boolean;
}

export const QueryHistory: React.FC<QueryHistoryProps> = ({ history, onSelectCity, loading }) => {
  const formatTime = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' +
             date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return isoString;
    }
  };

  return (
    <div className="glass-panel dashboard-card animate-fade-in" style={{ height: 'fit-content' }}>
      <h3 className="card-title">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 8v4l3 3"/><circle cx="12" cy="12" r="10"/></svg>
        Search History
      </h3>
      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
          <div className="spinner" style={{ width: '30px', height: '30px' }}></div>
        </div>
      ) : history.length === 0 ? (
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textAlign: 'center', padding: '1rem 0' }}>
          No previous searches found.
        </p>
      ) : (
        <div className="history-list">
          {history.map((query) => (
            <div
              key={query.id}
              className="history-item"
              onClick={() => onSelectCity(query.cityName.split(',')[0])}
            >
              <div className="history-header">
                <span className="history-city" title={query.cityName}>{query.cityName}</span>
                <span className="history-time">{formatTime(query.queriedAt)}</span>
              </div>
              <div className="history-details-row">
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  Lat: {query.latitude.toFixed(2)} Lon: {query.longitude.toFixed(2)}
                </span>
                {query.tempPredictedMean !== null && (
                  <span className="history-temp">
                    {query.tempPredictedMean.toFixed(1)}°C (ML)
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
