import java.io.*;

public class createTrain{
	public static void main(String[] args){
			File folder = new File("Annotations");	// Images
			// File folder = new File("../ImagesTest");   // Images
			File[] listOfFiles = folder.listFiles();
		
			//FileWriter fr = new FileWriter("ImageSets/train.txt");
			
			int pos;
			String name;
			for(int i = 0; i < listOfFiles.length; i++){
				if(listOfFiles[i].isFile()){
					name = listOfFiles[i].getName();
					pos = name.lastIndexOf(".");
					if (pos > 0) {
					    name = name.substring(0, pos);
					}
					//fr.write(name + "\n");
					System.out.println(name);
				}
			}
	}
}
