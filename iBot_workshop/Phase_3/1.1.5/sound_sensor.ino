const int soundPin = 3;
int sound = 0;

void setup() {
  pinMode(soundPin, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
}

void loop() {
  sound = digitalRead(soundPin);
  if(sound != 0){
    digitalWrite(LED_BUILTIN, HIGH);
    delay(2000);
    digitalWrite(LED_BUILTIN, LOW);
  }
}
