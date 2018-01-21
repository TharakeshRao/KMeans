package KClassifer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

/**
 * Training set for possible classes and domains of features
 * decision trees; contains instances
 * @author Alpha,Maya,Tharakesh
 * @param <T>
 *
 */
public class Trainingset<T> {

	private ArrayList<Instance<T>> instances;
	private ArrayList<Domain<T>> domains;
	private ArrayList<String> classes;
	
	public Trainingset(){
		instances = new ArrayList<Instance<T>>();
		domains = new ArrayList<Domain<T>>();
		classes = new ArrayList<String>();
	}
	
	public Trainingset(ArrayList<Domain<T>> domains, ArrayList<String> classes){
		this.domains = domains;
		this.classes = classes;
		instances = new ArrayList<Instance<T>>();
	}
	
	public void addInstance(Instance<T> i){
		instances.add(i);
	}
	
	public void setDomains(ArrayList<Domain<T>> domains){
		this.domains = domains;
	}
	
	public void setClasses(ArrayList<String> classes){
		this.classes = classes;
	}
	
	public ArrayList<String> getClasses(){
		return classes;
	}
	
	public ArrayList<Domain<T>> getDomains(){
		return domains;
	}
	
	public int getClassCount(){
		return classes.size();
	}
	
	public int getFeatureCount(){
		return domains.size();
	}
	
	public Domain<T> getDomain(int i){
		return domains.get(i);
	}
	
	public Instance<T> getInstance(int i){
		return instances.get(i);
	}
	
	public Instance<T> getRandomInstance(){
		return instances.get((int) (Math.random()*instances.size()));
	}
	
	public boolean isEmpty(){
		return instances.isEmpty();
	}
	
	public int size(){
		return instances.size();
	}
	
	/**
	 * Checks if training set only contains one class
	 * 
	 * @return if set is homogen
	 */
	public boolean isHomogen(){
		String classification = instances.get(0).getClassification();
		for(int i = 0; i < instances.size(); i++){
			if(classification.compareTo(instances.get(i).getClassification()) != 0)
				return false;
		}
		return true;
	}
	
	/**
	 * Computes the entropy of this training set
	 * 
	 * @param 
	 * classCount number of classes
	 * @return entropy
	 */
	public double getEntropy(int classCount){
		HashMap<String, Integer> classes = getClassMemberCount();
		
		double entropy = 0;
		Collection<Integer> values = classes.values();
		for(Integer value : values){
			double proportion = (double)(value)/(double)(instances.size());
			entropy += -1*proportion*Math.log(proportion)/Math.log(classCount);
		}
		
		return entropy;
	}
	
	
	public HashMap<String, Integer> getClassMemberCount() {
		HashMap<String,Integer> classes = new HashMap<String,Integer>();
		for(Instance<T> i : instances){
			if(classes.containsKey(i.getClassification())){
				classes.put(i.getClassification(), classes.get(i.getClassification()) + 1);
			}
			else
				classes.put(i.getClassification(), 1);
		}
		return classes;
	}

	
	public Trainingset<T> splitUpTestSet(int percentage){
		
		Trainingset<T> testset = new Trainingset<T>(domains, classes);
		int testsetsize =  (int) (instances.size()*percentage/(double) 100);
		
		for(int i = 0; i < testsetsize; i++){
			int index = (int)(Math.random()*instances.size());
			testset.addInstance(instances.remove(index));
		}
		
		return testset;
	}
	
}
