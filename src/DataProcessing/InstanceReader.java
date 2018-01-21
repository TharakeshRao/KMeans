package DataProcessing;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import KClassifer.Instance;
import KClassifer.Trainingset;

/**
 * Reads the instances of a training set from a file
 * 
 *  @author Alpha,Maya,Tharakesh
 *
 */
public class InstanceReader {

	String name;

	public InstanceReader(){
		
	}
	
	public InstanceReader(String name){
		this.name = name;
	}
	
	public void selectFile(String name) {
		this.name = name;
	}

	/**
	 * Reads instances from file
	 * 
	 * @return Training set of instances
	 */
	public Trainingset<String> readInstances() {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(name));
			
			Trainingset<String> res = new Trainingset<>();
			
			try {
				String str;
				while ((str = reader.readLine()) != null){
					Instance<String> i = stringToInstance(str);
					res.addInstance(i);
				}
				reader.close();
			} catch (IOException e) {
				System.err.println("Error while reading file");
				return null;
			}
			
			return res;
		} catch (IOException e) {
			System.err.println("No such file: " + name);
			return null;
		}
	}

	private Instance<String> stringToInstance(String str) {
		ArrayList<String> features = new ArrayList<String>(); 
		String[] split = str.split(",");
		for(int i = 0; i < split.length - 1; i++){
			features.add(split[i]);
		}
		Instance<String> i = new Instance<>(features, split[split.length - 1]);
		return i;
	}

}
