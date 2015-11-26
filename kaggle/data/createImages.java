import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import static java.nio.file.StandardCopyOption.*;

public class createImages{
	public static void main(String[] args){
		try{
			FileReader fr = new FileReader("ImageSets/train.txt");
			
			int pos;
			String name;
			
			BufferedReader br = new BufferedReader(fr);
			while (br.ready()) {
				name = br.readLine();
				Files.copy(Paths.get("Images/" + name + ".jpg"), 
						   Paths.get("Images2/" + name + ".jpg"), REPLACE_EXISTING);
			}

		}
		catch(IOException ex){
			ex.printStackTrace();
		}
	}
}
