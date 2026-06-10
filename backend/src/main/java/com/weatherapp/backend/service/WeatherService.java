package com.weatherapp.backend.service;

import com.weatherapp.backend.model.WeatherQuery;
import com.weatherapp.backend.repository.WeatherQueryRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class WeatherService {

    private final WeatherQueryRepository repository;
    private final RestTemplate restTemplate;

    @Value("${ml.service.url:http://localhost:8000}")
    private String mlServiceUrl;

    @Value("${openweather.api.key:YOUR_OPENWEATHER_API_KEY}")
    private String openWeatherApiKey;

    public WeatherService(WeatherQueryRepository repository) {
        this.repository = repository;
        this.restTemplate = new RestTemplate();
    }

    public Map<String, Object> geocodeCity(String cityName) {
        try {
            String geocodeUrl = "https://api.openweathermap.org/geo/1.0/direct?q=" + cityName + "&limit=1&appid=" + openWeatherApiKey;
            ResponseEntity<List> response = restTemplate.getForEntity(geocodeUrl, List.class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null && !response.getBody().isEmpty()) {
                Map<String, Object> result = (Map<String, Object>) response.getBody().get(0);
                Map<String, Object> locationData = new HashMap<>();
                locationData.put("latitude", result.get("lat"));
                locationData.put("longitude", result.get("lon"));
                locationData.put("name", result.get("name"));
                locationData.put("country", result.get("country"));
                return locationData;
            }
        } catch (Exception e) {
            System.err.println("Warning: OpenWeather geocoding failed, trying fallback to Open-Meteo: " + e.getMessage());
        }

        // Fallback to Open-Meteo Geocoding
        try {
            String openMeteoGeocodeUrl = "https://geocoding-api.open-meteo.com/v1/search?name=" + cityName + "&count=1&language=en&format=json";
            Map<String, Object> response = restTemplate.getForObject(openMeteoGeocodeUrl, Map.class);
            if (response != null && response.containsKey("results")) {
                List<Map<String, Object>> results = (List<Map<String, Object>>) response.get("results");
                if (results != null && !results.isEmpty()) {
                    Map<String, Object> result = results.get(0);
                    Map<String, Object> locationData = new HashMap<>();
                    locationData.put("latitude", result.get("latitude"));
                    locationData.put("longitude", result.get("longitude"));
                    locationData.put("name", result.get("name"));
                    locationData.put("country", result.get("country"));
                    return locationData;
                }
            }
        } catch (Exception e) {
            System.err.println("Error: Open-Meteo fallback geocoding failed: " + e.getMessage());
        }

        throw new RuntimeException("City not found or geocoding services unavailable: " + cityName);
    }

    private int mapOpenWeatherIdToWmoCode(int owId) {
        if (owId >= 200 && owId < 300) return 95; // Thunderstorm
        if (owId >= 300 && owId < 400) return 51; // Drizzle
        if (owId >= 500 && owId < 600) return 61; // Rain
        if (owId >= 600 && owId < 700) return 71; // Snow
        if (owId >= 700 && owId < 800) return 45; // Fog
        if (owId == 800) return 0;                // Clear
        if (owId > 800 && owId <= 804) return 3;  // Cloudy
        return 3;
    }

    public Map<String, Object> getForecastAndPredictions(String cityName) {
        // 1. Geocode city to get coordinates via OpenWeather
        Map<String, Object> locationData = geocodeCity(cityName);
        Double lat = ((Number) locationData.get("latitude")).doubleValue();
        Double lon = ((Number) locationData.get("longitude")).doubleValue();
        String resolvedName = (String) locationData.get("name") + ", " + (String) locationData.get("country");

        // 2. Fetch current weather from OpenWeather
        String openWeatherUrl = "https://api.openweathermap.org/data/2.5/weather?lat=" + lat + "&lon=" + lon + "&appid=" + openWeatherApiKey + "&units=metric";
        Map<String, Object> openWeatherData = null;
        try {
            openWeatherData = restTemplate.getForObject(openWeatherUrl, Map.class);
        } catch (Exception e) {
            System.err.println("Warning: Failed to fetch current weather from OpenWeather: " + e.getMessage());
        }

        // Map OpenWeather response to standard current_weather structure for Frontend compatibility
        Map<String, Object> weatherData = new HashMap<>();
        Map<String, Object> currentWeather = new HashMap<>();
        
        if (openWeatherData != null && openWeatherData.containsKey("main")) {
            Map<String, Object> main = (Map<String, Object>) openWeatherData.get("main");
            Map<String, Object> wind = (Map<String, Object>) openWeatherData.get("wind");
            List<Map<String, Object>> weatherList = (List<Map<String, Object>>) openWeatherData.get("weather");
            
            currentWeather.put("temperature", main.get("temp"));
            
            if (wind != null && wind.containsKey("speed")) {
                Double windMs = ((Number) wind.get("speed")).doubleValue();
                currentWeather.put("windspeed", Math.round(windMs * 3.6 * 10.0) / 10.0); // Convert m/s to km/h
            } else {
                currentWeather.put("windspeed", 0.0);
            }
            
            if (weatherList != null && !weatherList.isEmpty()) {
                Integer owId = ((Number) weatherList.get(0).get("id")).intValue();
                currentWeather.put("weathercode", mapOpenWeatherIdToWmoCode(owId));
            } else {
                currentWeather.put("weathercode", 0);
            }
            
            currentWeather.put("time", LocalDateTime.now().toString());
        } else {
            // Fallback to Open-Meteo Current Weather
            try {
                String openMeteoUrl = "https://api.open-meteo.com/v1/forecast?latitude=" + lat + "&longitude=" + lon + "&current_weather=true";
                Map<String, Object> openMeteoData = restTemplate.getForObject(openMeteoUrl, Map.class);
                if (openMeteoData != null && openMeteoData.containsKey("current_weather")) {
                    Map<String, Object> cw = (Map<String, Object>) openMeteoData.get("current_weather");
                    currentWeather.put("temperature", cw.get("temperature"));
                    currentWeather.put("windspeed", cw.get("windspeed"));
                    currentWeather.put("weathercode", cw.get("weathercode"));
                    currentWeather.put("time", cw.get("time"));
                } else {
                    throw new RuntimeException("Empty response from Open-Meteo current weather");
                }
            } catch (Exception e2) {
                System.err.println("Warning: Failed to fetch current weather from Open-Meteo fallback: " + e2.getMessage());
                currentWeather.put("temperature", 0.0);
                currentWeather.put("windspeed", 0.0);
                currentWeather.put("weathercode", 0);
                currentWeather.put("time", LocalDateTime.now().toString());
            }
        }
        weatherData.put("current_weather", currentWeather);

        // 3. Train / Make sure ML model is trained for these coordinates in ML service (still uses Open-Meteo archive)
        String trainUrl = mlServiceUrl + "/api/ml/train";
        Map<String, Object> trainRequest = new HashMap<>();
        trainRequest.put("latitude", lat);
        trainRequest.put("longitude", lon);
        try {
            restTemplate.postForObject(trainUrl, trainRequest, Map.class);
        } catch (Exception e) {
            System.err.println("Warning: ML model training failed: " + e.getMessage());
        }

        // 4. Fetch predictions from ML service
        String predictUrl = mlServiceUrl + "/api/ml/predict?latitude=" + lat + "&longitude=" + lon;
        Map<String, Object> mlPredictions = new HashMap<>();
        try {
            mlPredictions = restTemplate.getForObject(predictUrl, Map.class);
        } catch (Exception e) {
            System.err.println("Warning: Failed to fetch predictions: " + e.getMessage());
        }

        // 5. Save this query log in the database
        WeatherQuery query = new WeatherQuery();
        query.setCityName(resolvedName);
        query.setLatitude(lat);
        query.setLongitude(lon);
        query.setQueriedAt(LocalDateTime.now());
        
        Double predictedMean = null;
        if (mlPredictions != null && mlPredictions.containsKey("predictions")) {
            List<Map<String, Object>> preds = (List<Map<String, Object>>) mlPredictions.get("predictions");
            if (preds != null && !preds.isEmpty()) {
                predictedMean = ((Number) preds.get(0).get("ml_pred")).doubleValue();
            }
        }
        query.setTempPredictedMean(predictedMean);
        
        query.setDescription("Temp: " + currentWeather.get("temperature") + "°C, Wind: " + currentWeather.get("windspeed") + "km/h");
        
        repository.save(query);

        // 6. Merge and return results
        Map<String, Object> result = new HashMap<>();
        result.put("location", resolvedName);
        result.put("latitude", lat);
        result.put("longitude", lon);
        result.put("weather", weatherData);
        result.put("ml", mlPredictions);
        return result;
    }

    public byte[] getForecastChart(Double lat, Double lon) {
        String chartUrl = mlServiceUrl + "/api/ml/chart?latitude=" + lat + "&longitude=" + lon;
        try {
            return restTemplate.getForObject(chartUrl, byte[].class);
        } catch (Exception e) {
            throw new RuntimeException("Failed to fetch forecast chart from ML service", e);
        }
    }

    public List<WeatherQuery> getQueryHistory() {
        return repository.findAllByOrderByQueriedAtDesc();
    }
}
