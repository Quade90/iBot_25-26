#include <Servo.h>
Servo servo1;
int pos = 0;

void setup()
{
  servo1.attach(3);
  servo1.write(0);
}

void loop()
{
 	for(int i = 0; i<=180; i++){
    servo1.write(i);
    delay(10);
  }	
  for(int i = 180; i>=0; i--){
    servo1.write(i);
    delay(10);
  }
}