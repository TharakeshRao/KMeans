

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import KClassifer.Domain;
import KClassifer.KMean;
import KClassifer.Trainingset;
import KClassifer.Validator;
import DataProcessing.InstanceConverter;
import DataProcessing.InstanceReader;
import DataProcessing.MetaDataReader;

public class Main {

	public static void main(String[] args) {
		
		Integer[] a = new Integer[2];
		a[0] = new Integer(3);
		
		InstanceReader input = new InstanceReader("./car.data");
		Trainingset<String> t = input.readInstances();

		MetaDataReader meta = new MetaDataReader("./car.c45-names");
		ArrayList<Domain<String>> d = meta.readDomains();
		ArrayList<String> c = meta.readClasses();

		t.setClasses(c);
		t.setDomains(d);

		InstanceConverter converter = new InstanceConverter();
		Trainingset<Integer> newT = converter.convertStringToInteger(t);
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

		String kString = null;

		try {
			System.out.print("Enter k: ");
			kString = br.readLine();
		} catch (IOException e) {
			System.err.println("InputError occured");
			return;
		}
		
		Trainingset<Integer> testSet = null;
		Validator v = null;
		double mean = 0;
		
		for (int i = 0; i < 100; i++) {
			KMean classifier = new KMean(Integer.valueOf(kString));

			testSet = newT.splitUpTestSet(33);

			classifier.learn(newT);

			v = new Validator(classifier);
			
			mean += v.validateOnTestSet(testSet);
			
			newT = converter.convertStringToInteger(t);
		}
		System.out.println("Mean error over 100 samples: "
				+ (1-(mean/100.0)));

		int[][] confusion = v.computeConfusionMatrix(testSet);

		printConfusion(confusion, testSet.getClasses());

	}

	private static void printConfusion(int[][] confusion,
			ArrayList<String> classes) {
		System.out.print("\t | \t");
		for (int j = 0; j < confusion.length; j++)
			System.out.print(classes.get(j) + "\t | \t");
		System.out.println("");
		for (int i = 0; i < confusion.length; i++) {
			System.out.print(classes.get(i) + "  \t | \t");
			for (int j = 0; j < confusion.length; j++) {
				System.out.print(confusion[i][j] + "\t | \t");
			}
			System.out.println("");
		}
	}

}
