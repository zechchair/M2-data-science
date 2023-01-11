 //Define Class Language
 //Same comment as DepQues
class PWindow extends PApplet {
public boolean work1 = false;
ControlP5 cpExtra;
PApplet parent;
 
PWindow(PApplet app) {
super();
 PApplet.runSketch(new String[] {this.getClass().getSimpleName()},  this);
 
cpExtra = new ControlP5(this);
 
parent = app;


 PFont font = createFont("Times New Roman", 50);
 cpExtra.addButton("Francais")
.setPosition(1150,550)
.setSize(400,100)
.setFont(font)
.setColorLabel(255)
.setColorBackground(#595959)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
cpExtra.addButton("Anglais")
.setPosition(380,550)
.setSize(400,100)
.setFont(font)
.setColorLabel(255)
.setColorBackground(#595959)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
cpExtra.addButton("arabe")
.setPosition((380+1150)/2,800)
.setSize(400,100)
.setFont(font)
.setColorLabel(255)
.setColorBackground(#595959)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
}
 
void settings() {
fullScreen();
  x = 1800;
  o = 100;
  d = 100 ;
 
  smooth();
}
 


void draw(){

background(background_principal);
image(logo, 700, 0);
//image(robot, 50, 50);
//image(emineslogo2, 0, 0);

 charge();

 
}
 void texts(){
 textFont(fontt1);
textSize(50);

fill(#38517e);

 text("Choisissez la langue",720,350);

}
void texts1(){
 textFont(fontt1);
textSize(50);

fill(#ce3e57);

 text("Batterie faible",750,350);
 image(battery,1100,280);

}
void charge(){
   // show full circles
  ellipseMode(CENTER);
  fill(#ffc742);

  ellipse(x, o, d, d);//yellow1
  fill(#ffc742);
  ellipse(x, o, d-30, d-30);//green1
  fill(#ffc742);
   

 

  showArcs();
 
  // -----------------
  if (keyPressed&&key=='+') 
  
    charge++;
 
  if (keyPressed&&key=='-') 
    charge--;
 image(charging, 1780, 80,50,50);
  fill(255); // white
text_charge();

}
void text_charge(){    
  textFont(fontt, 20);
 fill(#000000);
 text(charge+"%",1785,105);}
void showArcs(){
  if (charge<15){
   myClient.write("1");
   
  fill(#FF0000);   // !!!!!!!!!!!!!!!!!!!!!!!!!

  arc(x, o, d, d, PI+HALF_PI, map(charge, 0, 75, PI+HALF_PI, PI+HALF_PI+PI+HALF_PI));//yellow  !!!!!!!!!!!!!!!   

 
  fill(255);
  arc(x, o, d-30, d-30, 0, TWO_PI);//center
  //booster = new UiBooster();
  //booster.showWarningDialog("Batterie faible ... ", "Warning");
  texts1();
   if (charge<=0){
    image(myAnimation2,  10, 10);}
    else{image(myAnimation1,  10, 10);}
}
  else{
     fill(#10E042);   // !!!!!!!!!!!!!!!!!!!!!!!!!
      
 
  arc(x, o, d, d, PI+HALF_PI, map(charge, 0, 75, PI+HALF_PI, PI+HALF_PI+PI+HALF_PI));//yellow  !!!!!!!!!!!!!!!   

 
  fill(255);
  arc(x, o, d-30, d-30, 0, TWO_PI);//center}

   image(myAnimation, 10, 10);
     texts();
 
  }
 
}


void Francais(){
 
myClient.write("fr");


delay(100);
language.hide();
}// send whatever you need to send here}
 

void Anglais(){
 myClient.write("en"); 

delay(100);
language.hide();}// send whatever you need to send here}}
 
 void arabe(){
 myClient.write("ar"); 

delay(100);
language.hide();}// send whatever you need to send here}}
 
void mouseReleased() {

}
public void show() {
 work1 = true;
 surface.setVisible(true);
 }
 public void hide() {
 work1 = false;
 surface.setVisible(false);
 }}
