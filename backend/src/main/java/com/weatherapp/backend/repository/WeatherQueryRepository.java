package com.weatherapp.backend.repository;

import com.weatherapp.backend.model.WeatherQuery;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface WeatherQueryRepository extends JpaRepository<WeatherQuery, Long> {
    List<WeatherQuery> findAllByOrderByQueriedAtDesc();
}
