package KClassifer;

/* @author Alpha,Maya,Tharakesh

*/
public class Validator {

	private IClassifier classifier;
	
	public Validator(IClassifier c){
		this.classifier = c;
	}
	
	public double validateOnTestSet(Trainingset<Integer> testSet){
		
		int wronglyClassified = 0;
		
		for(int i = 0; i < testSet.size(); i++){
			if(classifier.classify(testSet.getInstance(i)).compareTo(testSet.getInstance(i).getClassification()) != 0)
				wronglyClassified++;
		}
		
		return 1 - ((double) wronglyClassified  / (double) testSet.size());
	}
	
	public double validateOnTestSet(Trainingset<Integer> testSet, int samples){
		
		int wronglyClassified = 0;
		
		for(int i = 0; i < samples; i++){
			if(classifier.classify(testSet.getRandomInstance()).compareTo(testSet.getInstance(i).getClassification()) != 0)
				wronglyClassified++;
		}
		
		return 1 - ((double) wronglyClassified  / (double) samples);
	}
	
	public int[][] computeConfusionMatrix(Trainingset<Integer> testSet){
		
		int n = testSet.getClassCount();
		int[][] matrix = new int[n][n];
		
		for(int i = 0; i < testSet.size(); i++){
			int x = testSet.getClasses().indexOf(testSet.getInstance(i).getClassification());
			int y = testSet.getClasses().indexOf(classifier.classify(testSet.getInstance(i)));
			matrix[x][y]++;
		}
		
		return matrix;
	}
	
}
