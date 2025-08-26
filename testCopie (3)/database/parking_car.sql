CREATE DATABASE IF NOT EXISTS parking_db;
USE parking_db;

CREATE TABLE IF NOT EXISTS vehicle_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    phone VARCHAR(50) NOT NULL,
    license_plate VARCHAR(255) NOT NULL UNIQUE,
    arrival DATETIME NOT NULL,
    departure DATETIME NOT NULL,
    car_image VARCHAR(255) NOT NULL
) ENGINE=InnoDB;

