package DataProcessing;

import java.util.ArrayList;

import KClassifer.Domain;
import KClassifer.Instance;
import KClassifer.Trainingset;

public class InstanceConverter {

	public Trainingset<Integer> convertStringToInteger(Trainingset<String> org){
		
		ArrayList<Domain<Integer>> newDomains = convertDomains(org.getDomains());
		Trainingset<Integer> newSet = new Trainingset<>(newDomains, org.getClasses());
		
		for(int i = 0; i < org.size(); i++){
			
			ArrayList<Integer> newFeatures = new ArrayList<>();
			for(int f = 0; f < org.getInstance(i).getDimension(); f++){
				newFeatures.add(org.getDomain(f).indexOf(org.getInstance(i).getFeature(f)));
			}
			
			Instance<Integer> newInstance = new Instance<>(newFeatures, org.getInstance(i).getClassification());
			newSet.addInstance(newInstance);
		}
		
		return newSet;
	}
	
	private ArrayList<Domain<Integer>> convertDomains(ArrayList<Domain<String>> org){
		
		ArrayList<Domain<Integer>> newDomains = new ArrayList<>();
		
		for(Domain<String> d : org){
			Integer[] newVals = new Integer[d.size()];
			for(int i = 0; i < newVals.length; i++)
				newVals[i] = i;
			Domain<Integer> newDom = new Domain<>(d.getName(), newVals);
			newDomains.add(newDom);
		}
		
		return newDomains;
	}
	
}
