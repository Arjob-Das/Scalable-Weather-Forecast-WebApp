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

    @Value("${openweather.api.key:YOUR_OPENWEATHER_API_KEYYOUR_OPENWEATHER_API_KEY}")
    private String openWeatherApiKey;

    public WeatherService(WeatherQueryRepository repository) {
        this.repository = repository;
        this.restTemplate = new RestTemplate();
    }

    public Map<String, Object> geocodeCity(String cityName) {
        if (cityName == null || cityName.trim().isEmpty()) {
            throw new RuntimeException("City name cannot be empty");
        }

        // Clean and tokenize the input query
        String[] tokens = cityName.trim().split("[\\s,-]+");
        if (tokens.length == 0 || tokens[0].isEmpty()) {
            throw new RuntimeException("City name cannot be empty");
        }

        String primarySearchTerm = tokens[0];

        // Build regex pattern from tokens: token1.*token2.*token3
        StringBuilder regexPatternBuilder = new StringBuilder();
        for (int i = 0; i < tokens.length; i++) {
            if (i > 0) {
                regexPatternBuilder.append(".*");
            }
            regexPatternBuilder.append(java.util.regex.Pattern.quote(tokens[i].toLowerCase()));
        }
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(regexPatternBuilder.toString(), java.util.regex.Pattern.CASE_INSENSITIVE);

        // Try OpenWeather first
        try {
            String geocodeUrl = "https://api.openweathermap.org/geo/1.0/direct?q=" + primarySearchTerm + "&limit=10&appid=" + openWeatherApiKey;
            ResponseEntity<List> response = restTemplate.getForEntity(geocodeUrl, List.class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null && !response.getBody().isEmpty()) {
                List<Map<String, Object>> results = (List<Map<String, Object>>) response.getBody();
                Map<String, Object> bestMatch = findBestMatchOpenWeather(results, cityName, tokens[0], pattern);
                if (bestMatch != null) {
                    return bestMatch;
                }
            }
        } catch (Exception e) {
            System.err.println("Warning: OpenWeather geocoding failed, trying fallback to Open-Meteo: " + e.getMessage());
        }

        // Fallback to Open-Meteo Geocoding
        try {
            String openMeteoGeocodeUrl = "https://geocoding-api.open-meteo.com/v1/search?name=" + primarySearchTerm + "&count=20&language=en&format=json";
            Map<String, Object> response = restTemplate.getForObject(openMeteoGeocodeUrl, Map.class);
            if (response != null && response.containsKey("results")) {
                List<Map<String, Object>> results = (List<Map<String, Object>>) response.get("results");
                if (results != null && !results.isEmpty()) {
                    Map<String, Object> bestMatch = findBestMatchOpenMeteo(results, cityName, tokens[0], pattern);
                    if (bestMatch != null) {
                        return bestMatch;
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error: Open-Meteo fallback geocoding failed: " + e.getMessage());
        }

        throw new RuntimeException("City not found or geocoding services unavailable: " + cityName);
    }

    private Map<String, Object> findBestMatchOpenWeather(List<Map<String, Object>> results, String fullQuery, String firstToken, java.util.regex.Pattern pattern) {
        Map<String, Object> exactMatch = null;
        Map<String, Object> regexMatch = null;

        for (Map<String, Object> result : results) {
            String name = (String) result.get("name");
            String countryCode = (String) result.get("country");
            String state = (String) result.get("state");

            String countryName = countryCode;
            if (countryCode != null) {
                try {
                    countryName = new java.util.Locale("", countryCode).getDisplayCountry(java.util.Locale.US);
                } catch (Exception e) {
                    // Ignore
                }
            }

            String fullName = name + ", " + (state != null ? state + ", " : "") + countryName;

            // Check exact name match (case-insensitive) for the first token
            if (name != null && name.equalsIgnoreCase(firstToken)) {
                if (exactMatch == null) {
                    exactMatch = result;
                }
            }

            // Check regex match against the full name
            if (pattern.matcher(fullName).find()) {
                if (regexMatch == null) {
                    regexMatch = result;
                }
            }
        }

        Map<String, Object> selected = exactMatch != null ? exactMatch : (regexMatch != null ? regexMatch : results.get(0));

        Map<String, Object> locationData = new HashMap<>();
        locationData.put("latitude", ((Number) selected.get("lat")).doubleValue());
        locationData.put("longitude", ((Number) selected.get("lon")).doubleValue());
        locationData.put("name", selected.get("name"));
        locationData.put("country", selected.get("country"));
        return locationData;
    }

    private Map<String, Object> findBestMatchOpenMeteo(List<Map<String, Object>> results, String fullQuery, String firstToken, java.util.regex.Pattern pattern) {
        Map<String, Object> exactMatch = null;
        Map<String, Object> regexMatch = null;

        for (Map<String, Object> result : results) {
            String name = (String) result.get("name");
            String country = (String) result.get("country");
            String admin1 = (String) result.get("admin1");

            String fullName = name + ", " + (admin1 != null ? admin1 + ", " : "") + (country != null ? country : "");

            // Check exact name match (case-insensitive) for the first token
            if (name != null && name.equalsIgnoreCase(firstToken)) {
                if (exactMatch == null) {
                    exactMatch = result;
                }
            }

            // Check regex match against the full name
            if (pattern.matcher(fullName).find()) {
                if (regexMatch == null) {
                    regexMatch = result;
                }
            }
        }

        Map<String, Object> selected = exactMatch != null ? exactMatch : (regexMatch != null ? regexMatch : results.get(0));

        Map<String, Object> locationData = new HashMap<>();
        locationData.put("latitude", ((Number) selected.get("latitude")).doubleValue());
        locationData.put("longitude", ((Number) selected.get("longitude")).doubleValue());
        locationData.put("name", selected.get("name"));
        locationData.put("country", selected.get("country"));
        return locationData;
    }

    public List<Map<String, Object>> searchCities(String query) {
        if (query == null || query.trim().length() < 2) {
            return List.of();
        }

        String[] tokens = query.trim().split("[\\s,-]+");
        if (tokens.length == 0 || tokens[0].isEmpty()) {
            return List.of();
        }

        String primarySearchTerm = tokens[0];

        // Build regex pattern from tokens: token1.*token2.*token3
        StringBuilder regexPatternBuilder = new StringBuilder();
        for (int i = 0; i < tokens.length; i++) {
            if (i > 0) {
                regexPatternBuilder.append(".*");
            }
            regexPatternBuilder.append(java.util.regex.Pattern.quote(tokens[i].toLowerCase()));
        }
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(regexPatternBuilder.toString(), java.util.regex.Pattern.CASE_INSENSITIVE);

        List<Map<String, Object>> suggestions = new java.util.ArrayList<>();

        // Try OpenWeather first
        try {
            String geocodeUrl = "https://api.openweathermap.org/geo/1.0/direct?q=" + primarySearchTerm + "&limit=10&appid=" + openWeatherApiKey;
            ResponseEntity<List> response = restTemplate.getForEntity(geocodeUrl, List.class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                List<Map<String, Object>> results = (List<Map<String, Object>>) response.getBody();
                
                List<Map<String, Object>> exactMatches = new java.util.ArrayList<>();
                List<Map<String, Object>> otherMatches = new java.util.ArrayList<>();
                for (Map<String, Object> res : results) {
                    String name = (String) res.get("name");
                    String state = (String) res.get("state");
                    String countryCode = (String) res.get("country");
                    String countryName = countryCode;
                    if (countryCode != null) {
                        try {
                            countryName = new java.util.Locale("", countryCode).getDisplayCountry(java.util.Locale.US);
                        } catch (Exception e) {
                            // Ignore
                        }
                    }

                    String fullName = name + ", " + (state != null ? state + ", " : "") + countryName;
                    if (pattern.matcher(fullName).find()) {
                        Map<String, Object> sug = new HashMap<>();
                        sug.put("id", "ow-" + res.get("lat") + "-" + res.get("lon"));
                        sug.put("name", name);
                        sug.put("admin1", state);
                        sug.put("country", countryName);
                        sug.put("latitude", res.get("lat"));
                        sug.put("longitude", res.get("lon"));
                        
                        if (name != null && name.equalsIgnoreCase(tokens[0])) {
                            exactMatches.add(sug);
                        } else {
                            otherMatches.add(sug);
                        }
                    }
                }
                suggestions.addAll(exactMatches);
                suggestions.addAll(otherMatches);
            }
        } catch (Exception e) {
            System.err.println("Warning: OpenWeather search suggestions failed: " + e.getMessage());
        }

        // Try Open-Meteo to supplement or fallback
        if (suggestions.size() < 5) {
            try {
                String openMeteoGeocodeUrl = "https://geocoding-api.open-meteo.com/v1/search?name=" + primarySearchTerm + "&count=20&language=en&format=json";
                Map<String, Object> response = restTemplate.getForObject(openMeteoGeocodeUrl, Map.class);
                if (response != null && response.containsKey("results")) {
                    List<Map<String, Object>> results = (List<Map<String, Object>>) response.get("results");
                    if (results != null) {
                        List<Map<String, Object>> exactMatches = new java.util.ArrayList<>();
                        List<Map<String, Object>> otherMatches = new java.util.ArrayList<>();
                        for (Map<String, Object> res : results) {
                            String name = (String) res.get("name");
                            String admin1 = (String) res.get("admin1");
                            String country = (String) res.get("country");

                            String fullName = name + ", " + (admin1 != null ? admin1 + ", " : "") + (country != null ? country : "");
                            if (pattern.matcher(fullName).find()) {
                                boolean duplicate = false;
                                double lat = ((Number) res.get("latitude")).doubleValue();
                                double lon = ((Number) res.get("longitude")).doubleValue();
                                for (Map<String, Object> existing : suggestions) {
                                    if (existing.get("name").toString().equalsIgnoreCase(name) &&
                                        existing.get("country").toString().equalsIgnoreCase(country)) {
                                        duplicate = true;
                                        break;
                                    }
                                }
                                if (!duplicate) {
                                    Map<String, Object> sug = new HashMap<>();
                                    sug.put("id", "om-" + res.get("id"));
                                    sug.put("name", name);
                                    sug.put("admin1", admin1);
                                    sug.put("country", country);
                                    sug.put("latitude", lat);
                                    sug.put("longitude", lon);
                                    
                                    if (name != null && name.equalsIgnoreCase(tokens[0])) {
                                        exactMatches.add(sug);
                                    } else {
                                        otherMatches.add(sug);
                                    }
                                }
                            }
                        }
                        suggestions.addAll(exactMatches);
                        suggestions.addAll(otherMatches);
                    }
                }
            } catch (Exception e) {
                System.err.println("Warning: Open-Meteo search suggestions failed: " + e.getMessage());
            }
        }



        if (suggestions.size() > 5) {
            return suggestions.subList(0, 5);
        }
        return suggestions;
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
