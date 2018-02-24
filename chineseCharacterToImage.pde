PFont font;
String[] dataSimplified;
String[] dataTraditional;
int imageCount = 0;
void setup(){
  dataSimplified = loadStrings("simps.txt");
  dataTraditional = loadStrings("trads.txt");
  font = loadFont("SimSun-96.vlw");
  textFont(font,96);
  size(100,100);
  for(int line = 0; line < dataSimplified.length; line++){
    int ch = 0;
    //for(int ch = 0; ch < dataSimplified[line].length(); ch++){
      background(255);
      fill(0);
      text(dataSimplified[line].charAt(ch),2,84);
      saveFrame("simplified/"+imageCount+".png");
      background(255);
      fill(0);
      text(dataTraditional[line].charAt(ch),2,84);
      saveFrame("traditional/"+imageCount+".png");
      imageCount++;
      println(imageCount+" images are done.");
    //}
  }
}
void draw(){
}
