import React from 'react';

interface PredictionItem {
  date: string;
  temp_mean: number;
  humidity_mean: number;
  precipitation: number;
  ml_pred: number;
}

interface WeatherResponse {
  location: string;
  latitude: number;
  longitude: number;
  weather: {
    current_weather: {
      temperature: number;
      windspeed: number;
      weathercode: number;
      time: string;
    };
    daily: {
      time: string[];
      temperature_2m_max: number[];
      temperature_2m_min: number[];
      temperature_2m_mean: number[];
    };
  };
  ml?: {
    predictions?: PredictionItem[];
  };
}

interface WeatherDashboardProps {
  data: WeatherResponse;
  backendUrl: string;
  chartToken: number;
}

export const WeatherDashboard: React.FC<WeatherDashboardProps> = ({ data, backendUrl, chartToken }) => {
  const { location, latitude, longitude, weather, ml } = data;
  const current = weather.current_weather;
  const predictions = ml?.predictions || [];

  const getWeatherCodeText = (code: number) => {
    // Basic WMO Weather interpretation codes
    if (code === 0) return 'Clear Sky';
    if (code >= 1 && code <= 3) return 'Mainly Clear / Partly Cloudy';
    if (code === 45 || code === 48) return 'Foggy';
    if (code >= 51 && code <= 55) return 'Drizzle';
    if (code >= 61 && code <= 65) return 'Rainy';
    if (code >= 71 && code <= 75) return 'Snowy';
    if (code >= 80 && code <= 82) return 'Rain Showers';
    if (code >= 95 && code <= 99) return 'Thunderstorm';
    return 'Cloudy';
  };

  const formatDate = (dateStr: string) => {
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const date = new Date(dateStr);
    return {
      day: days[date.getDay()],
      dateLabel: date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    };
  };

  return (
    <div className="dashboard-col">
      {/* Current Weather Card */}
      <div className="glass-panel dashboard-card animate-fade-in">
        <div className="weather-hero">
          <div className="hero-main">
            <span className="coords">Latitude: {latitude.toFixed(4)} • Longitude: {longitude.toFixed(4)}</span>
            <h2 className="city-title">{location}</h2>
            <div className="temp-container">
              <span className="current-temp">{current.temperature.toFixed(1)}</span>
              <span className="temp-unit">°C</span>
            </div>
            <span className="weather-desc">{getWeatherCodeText(current.weathercode)}</span>
          </div>

          <div className="hero-details">
            <div className="weather-stat-card">
              <span className="stat-label">Wind Speed</span>
              <span className="stat-value">{current.windspeed} km/h</span>
            </div>
            <div className="weather-stat-card">
              <span className="stat-label">System Time</span>
              <span className="stat-value" style={{ fontSize: '0.85rem' }}>
                {new Date(current.time).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
            {predictions.length > 0 && (
              <>
                <div className="weather-stat-card">
                  <span className="stat-label">Humidity (Avg)</span>
                  <span className="stat-value">{predictions[0].humidity_mean.toFixed(0)}%</span>
                </div>
                <div className="weather-stat-card">
                  <span className="stat-label">Precipitation</span>
                  <span className="stat-value">{predictions[0].precipitation.toFixed(1)} mm</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Model Predictions & Comparison Chart Card */}
      <div className="glass-panel dashboard-card animate-fade-in" style={{ animationDelay: '0.1s' }}>
        <h3 className="card-title">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
          ML Forecast Model Analysis
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1rem' }}>
          Comparing predictions from the global pre-trained **PyTorch LSTM model** (which evaluates a 30-day sequence of historical daily weather) against the **Open-Meteo physical model forecast**.
        </p>

        <div className="chart-img-container">
          <img
            src={`${backendUrl}/api/weather/chart?latitude=${latitude}&longitude=${longitude}&t=${chartToken}`}
            className="chart-img"
            alt="ML Forecast Graph"
            onError={(e) => {
              // Fallback or retry
              (e.target as HTMLImageElement).style.display = 'none';
            }}
          />
        </div>
      </div>

      {/* Forecast Data Table Card */}
      {predictions.length > 0 && (
        <div className="glass-panel dashboard-card animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <h3 className="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3v19"/><path d="M5 12h14"/></svg>
            7-Day Forecast Comparison Table
          </h3>
          <div className="table-container">
            <table className="forecast-table">
              <thead>
                <tr>
                  <th>Day</th>
                  <th>Open-Meteo Temp</th>
                  <th>ML Predicted Temp</th>
                  <th>Variance</th>
                  <th>Avg Humidity</th>
                  <th>Precipitation</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((item, idx) => {
                  const dateInfo = formatDate(item.date);
                  const diff = item.ml_pred - item.temp_mean;
                  const diffClass = diff > 0.5 ? 'warmer' : diff < -0.5 ? 'cooler' : 'same';
                  const diffText = diff > 0.5 ? `+${diff.toFixed(1)}°C` : diff < -0.5 ? `${diff.toFixed(1)}°C` : '0.0°C';

                  return (
                    <tr key={idx}>
                      <td className="forecast-date">
                        <div>{dateInfo.day}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{dateInfo.dateLabel}</div>
                      </td>
                      <td className="forecast-api-temp">{item.temp_mean.toFixed(1)}°C</td>
                      <td className="forecast-ml-temp">{item.ml_pred.toFixed(1)}°C</td>
                      <td className={`forecast-diff ${diffClass}`}>
                        {diffText} {diffClass === 'warmer' ? '↑' : diffClass === 'cooler' ? '↓' : '•'}
                      </td>
                      <td>{item.humidity_mean.toFixed(0)}%</td>
                      <td>{item.precipitation.toFixed(1)} mm</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
