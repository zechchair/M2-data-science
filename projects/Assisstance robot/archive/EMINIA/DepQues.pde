  //Define Class Deplacement_Question
class DepQ extends PApplet {
 public boolean work = false;
ControlP5 cpExtra1;
PApplet parent;
 //Main void of the class 
DepQ(PApplet app1) {
super();
 PApplet.runSketch(new String[] {this.getClass().getSimpleName()},  this); 
cpExtra1 = new ControlP5(this);
parent = app1;
//build the buttons



}
 
void settings() {
fullScreen();
 
  x = 1800;
  o = 100;
  d = 100 ;
 
  smooth();
}
 
void setup(){

}

void draw(){
  //Desgn the window
  cursor();
background(background_principal);
image(logo, 700, 0);
image(quit,bX,bY);

//charge

 charge();

 if (list[1].equals("dq")){
 PFont font = createFont("Times New Roman", 50);
 cpExtra1.addButton("Deplacement")
.setPosition(1200,550)
.setSize(400,100)
.setFont(font)
.setColorLabel(255)
.setColorBackground(#595959)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
cpExtra1.addButton("Question")
.setPosition(350,550)
.setSize(400,100)
.setFont(font)
.setColorLabel(255)
.setColorBackground(#595959)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
cpExtra1.addButton("presenter les projets")
.setPosition(820,550)
.setSize(300,100)
.setFont(font)
.setColorLabel(255)
.setColorBackground(#595959)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);}
else if (list[1].equals("hand")){
  image(raise_hand,550,350,750,420);}
  else if (list[1].equals("maskfalse")){image(mask_required,570,300,800,470);}
  else if (list[1].equals("masktrue")){image(thanks,570,300,800,470);}
 
}
//Texts written
//1-
 void texts(){
 textFont(fontt1);
textSize(50);
fill(#38517e);
if (list[1].equals("dq")){
 text("Choisissez la fonctionalité souhaitée",560,300);}
 else if(list[1].equals("hand")){
text("S'il vous plaît levez votre main",560,300); }
}
//2-
void texts1(){
 textFont(fontt1);
textSize(50);

fill(#ce3e57);

 text("Batterie faible",750,300);
 image(battery,1100,280  );

}
//Void for determining the charge
//firs you can find in declaration in Eminia sketch that charge was set on a specific value that was read from python and vary every time
void charge(){
   // design of the icon charge
  ellipseMode(CENTER);
  fill(#ffc742);

  ellipse(x, o, d, d);//yellow1
  fill(#ffc742);
  ellipse(x, o, d-30, d-30);//green1
  fill(#ffc742);
   

  showArcs();
 //this part is just to make sure that charge coud vary
  // -----------------
  //Press + to add and -  to remove
  if (keyPressed&&key=='+') 
  
    charge++;
 
  if (keyPressed&&key=='-') 
    charge--;
 
 image(charging, 1780, 80,50,50);
  fill(255); // white
text_charge();

}
//text of percentage
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
 // Conditions on battery level
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
//Define what happen when we click to each button

void Deplacement(){
 
myClient.write("deplacement");

 
}// send whatever you need to send here}

void Question(){
 myClient.write("question");
 
}
void QRCODE(){
myClient.write("quitter");
 
}
void mouseReleased() {

}
public void show() {
 work = true;
 surface.setVisible(true);
 }
 public void hide() {
 work = false;
 surface.setVisible(false);
 }void mouseClicked() {
  if( mouseX > bX && mouseX < (bX + quit.width) &&
      mouseY > bY && mouseY < (bY + quit.height)){
       
         d_q.hide();
      };
    }
    void  mouseMoved() {
 if( mouseX > bX && mouseX < (bX + quit.width) &&
      mouseY > bY && mouseY < (bY + quit.height)){
         
        cursor(HAND);}
         
     
}
  }  
 
 
