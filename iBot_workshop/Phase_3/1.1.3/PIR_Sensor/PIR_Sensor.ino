const int PIRPin = 4;
int reading = 0;

void setup() {
  pinMode(PIRPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  delay(100);
  reading = digitalRead(PIRPin);
  Serial.println(reading);
}