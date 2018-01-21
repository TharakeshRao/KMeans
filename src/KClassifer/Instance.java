package KClassifer;
import java.util.ArrayList;


/**
 * Tuple of features and an optional classification
 * 
 * @author Alpha,Maya,Tharakesh
 * @param <T>
 *
 */
public class Instance<T> {

	private ArrayList<T> features;
	private String classification;
	
	public Instance(ArrayList<T> features, String classification){
		this.features = features;
		this.classification = classification;
	}
	
	public T getFeature(int index){
		return features.get(index);
	}
	
	public T removeFeature(int index){
		return features.remove(index);
	}
	
	public int getDimension(){
		return features.size();
	}
	
	public String getClassification(){
		return classification;
	}
	
	public String toString(){
		String res = "Instance( ";
		for(T s : features)
			res += s + ", ";
		res += ") class: " + classification;
		return res;
	}
	
}
