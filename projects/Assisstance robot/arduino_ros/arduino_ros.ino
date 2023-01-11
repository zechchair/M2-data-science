
#include <ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/Twist.h>
#include <ros/time.h>
//#include <FlexiTimer2.h>

#include <digitalWriteFast.h>

#define dir1 9
#define pwm1 6
#define encod1b 21
#define interrupt1b 2
#define encod1a 20
#define interrupt1a 3

#define dir2 4                            
#define pwm2 7
#define encod2b 19
#define interrupt2b 4
#define encod2a 18
#define interrupt2a 5

#define dir3 5
#define pwm3 8
#define encod3b 2
#define interrupt3b 0
#define encod3a 3
#define interrupt3a 1
#define T_echantillonnage 100
volatile double dt = T_echantillonnage/1000.;
float kp=0,kd=0,ki=0; 
float g=0;
char commande;
float x=0,tet=0;
double vx=0.0,vteta=0.0,vy=0.0,v1=0.0,v2=0.0,v3=0.0,vx_act=0.0,vy_act=0.0,vteta_act=0.0;
float PWM1=0,PWM2=0,PWM3=0;
int PWM = 240,cnt=2;
float R=27,r=5;
double pc1=0,pc2=0,pc3=0;
volatile long compteurImpulsions1 = 0,c1=1;
volatile long compteurImpulsions2 = 0,c2=1;
volatile long compteurImpulsions3 = 0,c3=1;
// Pour le moteur à courant continu:
unsigned long time;

float commande1=0,commande2=0,commande3=0;
float err1=0,err2=0,err3=0;
float errp1=0,errp2=0,errp3=0;


//volatile double dt = T_echantillonnage/1000.0;

//initializing all the variables
#define LOOPTIME      100     //Looptime in millisecond
int del=100;
int v=1000;
int ang;
int delay1;
int delay2;
const byte noCommLoopMax = 10;                //number of main loops the robot will execute without communication before stopping
unsigned int noCommLoops = 0;                 //main loop without communication counter

double speed_cmd_left2 = 0;      

float alpha1=0,alpha2=0,alpha3=0,alpha_m=0;
unsigned long lastMilli = 0;

//--- Robot-specific constants ---
const double radius = 0.1;                   //Wheel radius, in m
const double wheelbase = 0.47;               //Wheelbase, in m
const double encoder_cpr = 2080;               //Encoder ticks or counts per rotation
const double speed_to_pwm_ratio = 0.00235;    //Ratio to convert speed (in m/s) to PWM value. It was obtained by plotting the wheel speed in relation to the PWM motor command (the value is the slope of the linear function).
const double min_speed_cmd = 0.0882;          //(min_speed_cmd/speed_to_pwm_ratio) is the minimum command value needed for the motor to start moving. This value was obtained by plotting the wheel speed in relation to the PWM motor command (the value is the constant of the linear function).



                        
const double max_speed = 0.4;                 //Max speed in m/s

int PWM_leftMotor = 0;                     //PWM command for left motor
int PWM_rightMotor = 0;                    //PWM command for right motor 

volatile float pos_left = 0;       //Left motor encoder position
volatile float pos_right = 0;      //Right motor encoder position
ros::NodeHandle nh;

//function that will be called when receiving command from host
void handle_cmd (const geometry_msgs::Twist& cmd_vel) {
  noCommLoops = 0;                                                  //Reset the counter for number of main loops without communication
  
vx=cmd_vel.linear.x;
vy=0.0; 
vteta=cmd_vel.angular.z; 
v1=(vy+R*vteta/100)*(1/r*100*60/(30/PI)*15/10)  ;
  v2= (cos(2*PI/3)*vy-sin(2*PI/3)*vx+R/100*vteta)*(1/r*100*60/(30/PI)*30/20);
  v3=(cos(4*PI/3)*vy+R/100*vteta-sin(4*PI/3)*vx)*(1/r*100*60/(30/PI)*30/20);
     PWM1=map(v1,0,67,0,255);
     PWM2=map(v2,0,67,0,255);
     PWM3=map(v3,0,67,0,255);
}

ros::Subscriber<geometry_msgs::Twist> cmd_vel("cmd_vel", handle_cmd);   //create a subscriber to ROS topic for velocity commands (will execute "handle_cmd" function when receiving data)
geometry_msgs::Vector3Stamped speed_msg;                                //create a "speed_msg" ROS message
ros::Publisher speed_pub("speed", &speed_msg);  


void setup() {

          //prepare to publish speed in ROS topic

pinMode(encod1a,INPUT_PULLUP); pinMode(encod1b,INPUT_PULLUP);
pinMode(encod2a,INPUT_PULLUP); pinMode(encod2b,INPUT_PULLUP);
pinMode(encod3a,INPUT_PULLUP); pinMode(encod3b,INPUT_PULLUP);
Serial2.begin(9600);


pinMode(pwm1,OUTPUT); pinMode(pwm2,OUTPUT); pinMode(pwm3,OUTPUT);
pinMode(dir1,OUTPUT); pinMode(dir2,OUTPUT); pinMode(dir3,OUTPUT);

attachInterrupt(interrupt1a, GestionInterruptionCodeurPin1A, CHANGE);
attachInterrupt(interrupt1b, GestionInterruptionCodeurPin1B, CHANGE);
attachInterrupt(interrupt2a, GestionInterruptionCodeurPin2A, CHANGE);
attachInterrupt(interrupt2b, GestionInterruptionCodeurPin2B, CHANGE);
attachInterrupt(interrupt3a, GestionInterruptionCodeurPin3A, CHANGE);
attachInterrupt(interrupt3b, GestionInterruptionCodeurPin3B, CHANGE);
nh.initNode();                            //init ROS node
nh.getHardware()->setBaud(57600);         //set baud for ROS serial communication
nh.subscribe(cmd_vel);                    //suscribe to ROS topic for velocity commands
nh.advertise(speed_pub);         
// Pour compteur d'impulsions de l'encodeur:

//FlexiTimer2::set(T_echantillonnage, 1/1000., actual_speed);
//FlexiTimer2::start();

}

void loop() {

if (Serial2.available()>0){
  float cmd=Serial2.parseFloat();
 if (cmd==1234){

  float cmd1=Serial2.parseFloat();
  Serial2.read();
x=cmd1;
 equations(0,teta);
 delay(100);
 equations(x,0);

 
 }
else if (cmd==4321){

  float cmd1=Serial2.parseFloat();
  Serial2.read();


 
 }
else if(cmd==8888){
  Serial2.read();
  
 int sensorValue1 = analogRead(A0);
 delay(500);//read the A0 pin value
  int sensorValue2 = analogRead(A1);
  delay(500);//Serial.println(sensorValue);
  //float voltage = map(sensorValue,760,1000,10,12.6) ; //convert the value to a true voltage.
  int voltage1 = map(sensorValue1,679,733,51,98) ; //convert the value to a true voltage.
  int voltage2 = map(sensorValue2,711,728,92,99) ;  
int minn=min(voltage1,voltage2);
int maxx = max (voltage1, voltage2);
Serial2.println(minn);

 

}
}

nh.spinOnce();
if((millis()-lastMilli) >= LOOPTIME)   
{                                                                           // enter timed loop
lastMilli = millis();


pwm_cond(PWM1,PWM2,PWM3);
actual_speed();



if((millis()-lastMilli) >= LOOPTIME){         //write an error if execution time of the loop in longer than the specified looptime
Serial.println(" TOO LONG ");
}

noCommLoops++;
if (noCommLoops == 65535){
noCommLoops = noCommLoopMax;
}

publishSpeed(LOOPTIME);   //Publish odometry on ROS topic
}
}



//********************************************************************

 
void pwm_cond(float PWM1,float PWM2,float PWM3){
if(PWM1<=0) {
  digitalWrite(dir1,LOW);
 }
 else if(PWM1>0) {
  digitalWrite(dir1,HIGH);
 }
  if(PWM2<=0) {
  digitalWrite(dir2,LOW);
 }
 else if(PWM2>0) {
  digitalWrite(dir2,HIGH);
 }
  if(PWM3<=0) {
  digitalWrite(dir3,LOW);
 }
 else if(PWM3>0) {
  digitalWrite(dir3,HIGH);
 }
 if(PWM1>255) {
  PWM1 = 255;
 }
  if(PWM2>255) {
  PWM2 = 255;
 }
  if(PWM3>255) {
  PWM3 = 255;
 }
 if(PWM1<-255) {
  PWM1 = -255;
 }
  if(PWM2<-255) {
  PWM2 = -255;
 }
  if(PWM3<-255) {
  PWM3 = -255;
 }
 int pwm11=(int)PWM1;
 int pwm22=(int)PWM2;
 int pwm33=(int)PWM3;

 analogWrite(pwm1,abs(pwm11));
analogWrite(pwm2,abs(pwm22));

analogWrite(pwm3,abs(pwm33));
}



void equations(float x ,float teta){
alpha1=0.0;   alpha2=0.0;  alpha3=0.0;  alpha_m=0.0;
commande1=-(R*teta*2*PI)/(2*PI*r)  ;
commande2= -(-sin(4*PI/3)*x+R*teta*2*PI)/(2*PI*r); //vx en cm /s vteta tr/s
commande3=-(-sin(2*PI/3)*x+R*teta*2*PI)/(2*PI*r);
//Serial.print(commande1);Serial.print("ccccc");Serial.print(commande2);Serial.print("ccccc");
//Serial.print(commande3);Serial.println("ccccc");
c1=commande1;  c2=commande2;  c3=commande3;
float maxx=max(max(abs(commande1),abs(commande2)),abs(commande3));

PWM1=PWM*commande1/maxx;PWM2=PWM*commande2/maxx;PWM3=PWM*commande3/maxx;

  go();

}







void go() {
 
if(commande1!=0.0){alpha1=abs(c1)/(commande1*2080);}
if(commande2!=0.0){alpha2=abs(c2)/(commande2*2080);}
if(commande3!=0.0){alpha3=abs(c3)/(commande3*2080);}

while(abs(alpha1)<=1.0 && abs(alpha2)<=1.0 && abs(alpha3)<=1.0){
 
if(commande1!=0.0){alpha1=abs(c1)/(commande1*2080);}
if(commande2!=0.0){alpha2=abs(c2)/(commande2*2080);}
if(commande3!=0.0){alpha3=abs(c3)/(commande3*2080);}


if(commande1!=0.0 && commande2!=0.0 &&commande3!=0.0){alpha_m=(abs(alpha2)+abs(alpha1)+abs(alpha3))/3.0;}
else if(commande1==0.0){alpha_m=(abs(alpha2)+abs(alpha3))/2.0;}
else if(commande2==0.0){alpha_m=(abs(alpha1)+abs(alpha3))/2.0;}
else if(commande3==0.0){alpha_m=(abs(alpha2)+abs(alpha1))/2.0;}
err1=(alpha_m-abs(alpha1));
err2=(alpha_m-abs(alpha2));
err3=(alpha_m-abs(alpha3));

Serial2.print(alpha1);Serial2.print("********");Serial2.print(alpha2);Serial2.print("********");Serial2.print(alpha3);Serial2.println("********");
if(abs(alpha1)<=1.0 && commande1>0.0){PWM1+=kp*err1;} else if(abs(alpha1)<=1.0 && commande1<0.0){PWM1-=kp*err1;}
if(abs(alpha2)<=1.0 && commande2>0.0){PWM2+=kp*err2;} else if(abs(alpha2)<=1.0 && commande2<0.0){PWM2-=kp*err2;}
if(abs(alpha3)<=1.0 && commande3>0.0){PWM3+=kp*err3;} else if(abs(alpha3)<=1.0 && commande3<0.0){PWM3-=kp*err3;}
errp1=err1;
errp2=err2;
errp3=err3;
delay(10);
pwm_cond(PWM1,PWM2,PWM3);}
PWM1=0;PWM2=0;PWM3=0;
pwm_cond(PWM1,PWM2,PWM3);
}




















//*************************************************************************************************


void actual_speed(){

   pc1 = ((double)compteurImpulsions1*60)/(2080.0*dt);  // Pour la vitesse de sortie en tr/min
    pc2 = ((double)compteurImpulsions2*60)/(2080.0*dt);  
    pc3 = ((double)compteurImpulsions3*60)/(2080.0*dt);
   
    vx_act=(-sqrt(3)*pc2/3+sqrt(3)*pc3/3)/(1/r*100*60/(30/PI)*30/20);
    vy_act=(2*pc1/3-pc2/3-pc3/3)/(1/r*100*60/(30/PI)*30/20);
    vteta_act=((pc1+pc2+pc3)*4/3)/(1/r*100*60/(30/PI)*30/20);
    compteurImpulsions1=0;compteurImpulsions2=0;compteurImpulsions3=0;

}






   
// Pour la routine de service d'interruption attachée à la voie A du codeur incrémental:
void GestionInterruptionCodeurPin1A()
    {
     if (digitalReadFast2(encod1a) == digitalReadFast2(encod1b))
          {compteurImpulsions3++;c3++;}
     else {compteurImpulsions3--;c3--;}
    }
 
// Pour la routine de service d'interruption attachée à la voie B du codeur incrémental:
void GestionInterruptionCodeurPin1B()
    {
     if (digitalReadFast2(encod1a) == digitalReadFast2(encod1b))
          {compteurImpulsions3--;c3--;}
     else {compteurImpulsions3++;c3++;}
    }



void GestionInterruptionCodeurPin2A()
    {
     if (digitalReadFast2(encod2a) == digitalReadFast2(encod2b))
          {compteurImpulsions1++;c1++;}
     else {compteurImpulsions1--;c1--;}
    }
 
// Pour la routine de service d'interruption attachée à la voie B du codeur incrémental:
void GestionInterruptionCodeurPin2B()
    {
     if (digitalReadFast2(encod2a) == digitalReadFast2(encod2b))
          {compteurImpulsions1--;c1--;}
     else {compteurImpulsions1++;c1++;}
    }
    void GestionInterruptionCodeurPin3A()
    {
     if (digitalReadFast2(encod3a) == digitalReadFast2(encod3b))
          {compteurImpulsions2++;c2++;}
     else {compteurImpulsions2--;c2--;}
    }
 
// Pour la routine de service d'interruption attachée à la voie B du codeur incrémental:
void GestionInterruptionCodeurPin3B()
    {
     if (digitalReadFast2(encod3a) == digitalReadFast2(encod3b))
          {compteurImpulsions2--;c2--;}
     else {compteurImpulsions2++;c2++;}
    }




//Publish function for odometry, uses a vector type message to send the data (message type is not meant for that but that's easier than creating a specific message type)
void publishSpeed(double time) {
  speed_msg.header.stamp = nh.now();      //timestamp for odometry data
  speed_msg.vector.x = vx_act;    //left wheel speed (in m/s)
  speed_msg.vector.y = vteta_act;   //right wheel speed (in m/s)
  speed_msg.vector.z = time/1000;         //looptime, should be the same as specified in LOOPTIME (in s)
  speed_pub.publish(&speed_msg);
  nh.spinOnce();
  nh.loginfo("Publishing odometry");
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}
