const int PIN_GREEN_LED = 5;
const int PIN_RED_LED   = 6;

void setup() {
  pinMode(PIN_GREEN_LED, OUTPUT);
  pinMode(PIN_RED_LED, OUTPUT);
  digitalWrite(PIN_GREEN_LED, LOW);
  digitalWrite(PIN_RED_LED, LOW);
  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    if (c == '1') {
      digitalWrite(PIN_GREEN_LED, HIGH); 
      digitalWrite(PIN_RED_LED, LOW);
    } 
    else if (c == '0') {
      digitalWrite(PIN_GREEN_LED, LOW);
      digitalWrite(PIN_RED_LED, HIGH);  
    } 
    else if (c == '2') {
      digitalWrite(PIN_GREEN_LED, LOW);
      digitalWrite(PIN_RED_LED, LOW);
    }
  }
}
