const int IRPin = A0;
int reading = 0;

void setup() {
  pinMode(IRPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  delay(100);
  reading = analogRead(IRPin);
  Serial.println(reading);
}
