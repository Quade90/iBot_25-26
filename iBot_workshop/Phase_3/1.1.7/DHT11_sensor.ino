#include <DHT.h>
const int DHT11_pin=4;

float Humidity,Temperature;
DHT my_DHT11(DHT11_pin,DHT11);

void setup() {
  my_DHT11.begin();
  Serial.begin(9600);
}
void loop() {
 Humidity = my_DHT11.readHumidity();
 Temperature = my_DHT11.readTemperature();

  if(isnan(Humidity) || isnan(Temperature))
  {
    Serial.println("Reading Failed");
  }
  else
  {
    Serial.print("Temp: ");
    Serial.print(Temperature);
    Serial.println(" C");

    Serial.print("Humidity: ");
    Serial.print(Humidity);
    Serial.println(" %");
  }
 delay(2000);
}